from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .. import paths
from .pillar import FEAT_DIM, GLOBAL_DIM, PillarFeatureExtractor, PillarFeatures

TEXT_PROJ_KEYS = ("model.text.proj.0.weight", "model.text.proj.2.weight")
# 재정렬 encoder는 patch_embed 캐시(znorm=False)로 학습되므로 이후 캐싱도 반드시 znorm=False.
ENCODER_ZNORM = False


class ContrastiveAligner(nn.Module):
    def __init__(self, n_presence: int, pillar_path: Optional[str] = None,
                 freeze_patch_embed: bool = True, amp_dtype=torch.bfloat16,
                 device: str = "cuda"):
        super().__init__()
        self.enc = PillarFeatureExtractor(pillar_path or None, device=device, trainable=True,
                                          freeze_patch_embed=freeze_patch_embed, amp_dtype=amp_dtype)
        sd = load_file(str(Path(pillar_path or paths.PILLAR0) / "model.safetensors"))
        self.text_proj = nn.Sequential(nn.Linear(4096, 2624, bias=False), nn.GELU(),
                                       nn.Linear(2624, GLOBAL_DIM, bias=False))
        self.text_proj[0].weight.data.copy_(sd[TEXT_PROJ_KEYS[0]])
        self.text_proj[2].weight.data.copy_(sd[TEXT_PROJ_KEYS[1]])
        self.text_proj.requires_grad_(False)                       # 고정 anchor

        self.local_img_proj = nn.Sequential(nn.LayerNorm(FEAT_DIM), nn.Linear(FEAT_DIM, GLOBAL_DIM))
        self.n_presence = n_presence
        self.presence_head = nn.Sequential(nn.LayerNorm(GLOBAL_DIM), nn.Linear(GLOBAL_DIM, n_presence))

        # calibration (SigLIP scale/bias, local InfoNCE scale) — Pillar-0 체크포인트 값에서 시작.
        self.logit_scale = nn.Parameter(torch.tensor(3.0464))       # exp ≈ 21.04
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.logit_scale_local = nn.Parameter(torch.tensor(2.3026))  # exp ≈ 10
        self.to(device)

    def _outputs(self, feats: PillarFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """PillarFeatures → img_g (B,1152) L2, img_p (B,512,1152) L2 (grids[1]=8³)."""
        img_g = F.normalize(feats.global_embed.float(), dim=-1)
        g1 = feats.grids[1]                                          # (B,8,8,8,384)
        img_p = F.normalize(self.local_img_proj(g1.reshape(g1.size(0), -1, FEAT_DIM).float()), dim=-1)
        return img_g, img_p                                          # (B,1152), (B,512,1152)

    def encode_image_cached(self, pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """캐시된 patch_embed 출력 (B,384,32,32,32) → (img_g, img_p)."""
        return self._outputs(self.enc.forward_cached(pe))

    def presence_logits(self, img_g: torch.Tensor) -> torch.Tensor:
        return self.presence_head(img_g)

    def text_feat(self, qwen4096: torch.Tensor) -> torch.Tensor:
        """(N,4096) Qwen3 임베딩 → (N,1152) L2 (frozen 투영)."""
        return F.normalize(self.text_proj(qwen4096.float()), dim=-1)

    def scales(self):
        return (self.logit_scale.exp().clamp(max=100.0), self.logit_bias,
                self.logit_scale_local.exp().clamp(max=100.0))

    def trainable_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """(encoder params, head params) — differential LR용."""
        enc_p = [p for p in self.enc.parameters() if p.requires_grad]
        head_p = (list(self.local_img_proj.parameters()) + list(self.presence_head.parameters())
                  + [self.logit_scale, self.logit_bias, self.logit_scale_local])
        return enc_p, head_p


def export_encoder(aligner_ckpt: Path, out_path: Path) -> int:
    sd = torch.load(aligner_ckpt, map_location="cpu", weights_only=False)["aligner"]
    enc = {k[len("enc."):]: v for k, v in sd.items() if k.startswith("enc.")}
    assert enc, f"{aligner_ckpt}에 enc.* 키가 없다"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": enc, "znorm": ENCODER_ZNORM, "source": str(aligner_ckpt)}, out_path)
    return len(enc)


def load_encoder_ckpt(path: Path) -> Tuple[dict, bool]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], bool(obj.get("znorm", ENCODER_ZNORM))
    return obj, ENCODER_ZNORM
