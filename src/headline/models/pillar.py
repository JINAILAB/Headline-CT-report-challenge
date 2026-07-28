from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .. import paths

# 스케일별 (window grid dw,hw,ww) 와 within-window (m0,m1,m2)
# scale0: grid 8³ × win 4³ = 32³ | scale1: grid 2³ × win 4³ = 8³ | scale2: grid 2³ (window 없음)
_SCALE_LAYOUT = [
    dict(dw=8, hw=8, ww=8, m0=4, m1=4, m2=4),
    dict(dw=2, hw=2, ww=2, m0=4, m1=4, m2=4),
    dict(dw=2, hw=2, ww=2, m0=1, m1=1, m2=1),
]
SCALE_GRIDS = [(32, 32, 32), (8, 8, 8), (2, 2, 2)]   # (D,H,W) per scale
FEAT_DIM = 384          # 스케일별 채널
GLOBAL_DIM = 1152       # 3×384 = Pillar CLIP embed dim
PE_SHAPE = (FEAT_DIM, 32, 32, 32)   # patch_embed 캐시 shape


def unwindow_feats(feats_for_head: List[torch.Tensor]) -> List[torch.Tensor]:
    """[(B,N,384)×3] → [(B,D,H,W,384)×3] (32³/8³/2³). 토큰순서 (dw hw ww m0 m1 m2)."""
    grids = []
    for i, f in enumerate(feats_for_head):
        layout = _SCALE_LAYOUT[i]
        if layout["m0"] == 1:                       # scale2: window 없음
            g = rearrange(f, "b (dw hw ww) c -> b dw hw ww c",
                          dw=layout["dw"], hw=layout["hw"], ww=layout["ww"])
        else:
            g = rearrange(f, "b (dw hw ww m0 m1 m2) c -> b (dw m0) (hw m1) (ww m2) c", **layout)
        grids.append(g)                             # (B,D,H,W,384)
    return grids


@dataclass
class PillarFeatures:
    grids: List[torch.Tensor]      # [(B,D,H,W,384)] scale0/1/2 = 32³/8³/2³
    global_embed: torch.Tensor     # (B,1152) L2-normalised Pillar CLIP embedding


class _CachedPatchEmbed(nn.Module):
    """forward_cached 전용: 입력을 무시하고 캐시된 patch_embed 출력을 반환."""

    def __init__(self, pe: torch.Tensor):
        super().__init__()
        self.pe = pe

    def forward(self, x):                            # noqa: D102 - 입력 무시가 의도
        return self.pe


class PillarFeatureExtractor(nn.Module):
    """Pillar-0 vision tower wrapper.

    trainable=False(기본): frozen. VLM과 feature 캐싱이 이 모드를 쓴다.
    trainable=True        : visual tower unfreeze — contrastive 재정렬 학습이 이 모드를 쓴다.
                            conv stem(patch_embed)은 freeze해 activation 메모리를 크게 줄인다.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda",
                 dtype: torch.dtype = torch.float32, *, trainable: bool = False,
                 freeze_patch_embed: bool = True, amp_dtype: Optional[torch.dtype] = None):
        # 파라미터는 fp32 유지: 내부 posemb(cpb_mlp)가 fp32라 .to(bf16)이면 충돌한다.
        # bf16은 autocast(amp_dtype)로만 쓴다.
        super().__init__()
        import sys
        from pathlib import Path

        mp = Path(model_path or paths.PILLAR0).resolve()
        # trust_remote_code 동적모듈 캐시 버그 우회: relative-import 패키지를 직접 로드한다.
        parent, pkg = str(mp.parent), mp.name
        if parent not in sys.path:
            sys.path.insert(0, parent)
        mod = __import__(f"{pkg}.modeling_clip_mmatlas", fromlist=["CLIPMultimodalAtlas"])
        self.model = mod.CLIPMultimodalAtlas.from_pretrained(str(mp))
        self.model.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.visual = self.model.model.visual          # MultiModalAtlas — hook attr 보유
        self.trainable, self.amp_dtype = trainable, amp_dtype
        if trainable:
            self.model.requires_grad_(False)
            self.visual.requires_grad_(True)
            self.model.model.text.requires_grad_(False)     # text tower는 encode_image에서 미사용
            if freeze_patch_embed:
                self.visual.patch_embeds.requires_grad_(False)
            self.train()
        else:
            self.model.eval()
            self.model.requires_grad_(False)

    def _freeze_bn(self):
        # 작은 3D 배치에서 BN running stat이 흔들리지 않게 고정.
        for m in self.visual.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "trainable", False):
            self._freeze_bn()
        return self

    def _contexts(self):
        grad_ctx = nullcontext() if self.trainable else torch.no_grad()
        amp_ctx = (torch.autocast("cuda", dtype=self.amp_dtype)
                   if self.amp_dtype is not None else nullcontext())
        return grad_ctx, amp_ctx

    def _check(self, grids: List[torch.Tensor]) -> None:
        for g, (D, H, W) in zip(grids, SCALE_GRIDS):
            assert g.shape[1:] == (D, H, W, FEAT_DIM), (g.shape, (D, H, W, FEAT_DIM))

    def forward(self, brain_ct: torch.Tensor) -> PillarFeatures:
        # brain_ct: (B,11,128,256,256)
        assert brain_ct.dim() == 5 and brain_ct.size(1) == 11, brain_ct.shape
        x = brain_ct.to(device=self.device, dtype=self.dtype)
        grad_ctx, amp_ctx = self._contexts()
        with grad_ctx, amp_ctx:
            emb = self.model.extract_vision_feats(image={"brain_ct": x})   # (B,1152), hook 채움
            feats = self.visual._last_feats_for_head                        # [(B,N,384)×3]
            grids = unwindow_feats([f.float() for f in feats])              # [(B,D,H,W,384)]
        self._check(grids)
        return PillarFeatures(grids=grids, global_embed=emb.float())

    def forward_cached(self, pe: torch.Tensor) -> PillarFeatures:
        """캐시된 frozen patch_embed 출력에서 forward. pe: (B,384,32,32,32).
        """
        assert pe.dim() == 5 and tuple(pe.shape[1:]) == PE_SHAPE, pe.shape
        pe = pe.to(device=self.device, dtype=self.dtype)
        key = "brain_ct"
        orig = self.visual.patch_embeds[key]
        self.visual.patch_embeds[key] = _CachedPatchEmbed(pe)
        dummy = pe.new_empty(pe.shape[0], 1)          # forward()는 image.shape[0]=B만 참조한다
        grad_ctx, amp_ctx = self._contexts()
        try:
            with grad_ctx, amp_ctx:
                emb = self.model.extract_vision_feats(image={key: dummy})   # (B,1152)
                feats = self.visual._last_feats_for_head
                grids = unwindow_feats([f.float() for f in feats])
        finally:
            self.visual.patch_embeds[key] = orig
        self._check(grids)
        return PillarFeatures(grids=grids, global_embed=emb.float())
