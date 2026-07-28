from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Union

import torch
import torch.nn as nn

from .. import paths
from .pillar import PillarFeatureExtractor
from .tokenizers import GlobalTokenizer, RegionTokenizer

PostIds = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass
class VLMConfig:
    qwen_path: str = ""                       # 빈 값이면 paths.QWEN_LLM
    pillar_path: str = ""                     # 빈 값이면 paths.PILLAR0
    num_global_queries: int = 48
    region_dropout_p: float = 0.5             # 학습 시 region 블록을 통째로 뺄 확률
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    train_llm: bool = False                   # stage1=False, stage2=True(LoRA)
    load_pillar: bool = False                 # 캐시 학습이 기본이라 False
    grad_ckpt: bool = False                   # activation 재계산(메모리↓, throughput↓)
    gen_attn: str = "sdpa"                    # 생성 동안만 쓰는 attention 구현.
    # flash-attn 2.8.x가 생성 중 CUDA illegal-memory-access를 내는 케이스가 있어 생성만 sdpa로 바꾼다.

    def resolved_qwen(self) -> str:
        return self.qwen_path or str(paths.QWEN_LLM)


class PillarVLM(nn.Module):
    def __init__(self, cfg: VLMConfig, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = device
        qwen = cfg.resolved_qwen()
        self.tok = AutoTokenizer.from_pretrained(qwen)

        def _load(attn: str):
            return AutoModelForCausalLM.from_pretrained(
                qwen, torch_dtype=torch.bfloat16, attn_implementation=attn)

        try:                                        # flash-attn 있으면 사용, 없으면 sdpa
            self.llm = _load("flash_attention_2")
        except Exception:                           # noqa: BLE001 — 환경 차이 흡수
            self.llm = _load("sdpa")
        self.llm.config.use_cache = False           # 학습 중 KV 캐시 불필요
        H = self.llm.config.hidden_size             # 3584 (Qwen2.5-7B)
        self.H = H
        print(f"[PillarVLM] attn={self.llm.config._attn_implementation} hidden={H}", flush=True)

        self.pillar = (PillarFeatureExtractor(cfg.pillar_path or None, device=device)
                       if cfg.load_pillar else None)
        self.global_tok = GlobalTokenizer(out_dim=H, num_queries=cfg.num_global_queries)
        self.region_tok = RegionTokenizer(out_dim=H)

        for p in self.llm.parameters():
            p.requires_grad_(False)
        if cfg.grad_ckpt:
            self.llm.gradient_checkpointing_enable()
        if cfg.train_llm:
            from peft import LoraConfig, get_peft_model
            self.llm = get_peft_model(self.llm, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_targets, task_type="CAUSAL_LM"))
        self.to(device)

    # ── helpers ───────────────────────────────────────────────────────────────
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids.to(self.device))

    def trainable_state_dict(self) -> dict:
        """frozen Pillar/Qwen을 뺀 학습 대상만 저장 — visual tokenizer + LoRA."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("global_tok", "region_tok")) or "lora_" in k.lower()}

    def _use_region(self, batch_size: int, region: str) -> torch.Tensor:
        if region == "dropout":                                     # 학습
            return torch.rand(batch_size) > self.cfg.region_dropout_p
        if region == "on":
            return torch.ones(batch_size, dtype=torch.bool)
        if region == "off":                                         # 보고된 생성 설정
            return torch.zeros(batch_size, dtype=torch.bool)
        raise ValueError(f"region must be dropout|on|off, got {region!r}")

    def encode_vision(self, region_masks, use_region: torch.Tensor,
                      brain_ct=None, cached=None) -> List[torch.Tensor]:
        """→ 샘플별 (Nvis_i, H) 리스트. cached=(grids, global_embed)면 Pillar forward를 건너뛴다."""
        if cached is not None:
            grids = [g.float().to(self.device) for g in cached[0]]   # [(B,D,H,W,384)]
            global_embed = cached[1].float().to(self.device)         # (B,1152)
        else:
            feats = self.pillar(brain_ct)
            grids = [g.to(self.device) for g in feats.grids]
            global_embed = feats.global_embed.to(self.device)
        gtok = self.global_tok(grids, global_embed)                  # (B,Q+1,H)
        rtok = self.region_tok(grids, region_masks.to(self.device).float())   # (B,R,H)
        assert gtok.size(1) == self.cfg.num_global_queries + 1, gtok.shape
        return [torch.cat([gtok[b], rtok[b]], 0) if use_region[b] else gtok[b]
                for b in range(gtok.size(0))]

    @staticmethod
    def _post(post_ids: PostIds, b: int) -> torch.Tensor:
        return post_ids[b] if isinstance(post_ids, (list, tuple)) else post_ids

    # ── train / eval ──────────────────────────────────────────────────────────
    def forward(self, region_masks, pre_ids: List[torch.Tensor], post_ids: PostIds,
                report_ids: List[torch.Tensor], region: str = "dropout",
                cached=None, brain_ct=None) -> torch.Tensor:
        B = len(pre_ids)
        vis = self.encode_vision(region_masks, self._use_region(B, region),
                                 brain_ct=brain_ct, cached=cached)
        seqs, labels = [], []
        for b in range(B):
            pre_e = self.embed(pre_ids[b])
            post_e = self.embed(self._post(post_ids, b))
            rep_e = self.embed(report_ids[b])
            seq = torch.cat([pre_e, vis[b].to(pre_e.dtype), post_e, rep_e], 0)   # (L,H)
            lab = torch.full((seq.size(0),), -100, dtype=torch.long, device=self.device)
            lab[-report_ids[b].size(0):] = report_ids[b].to(self.device)         # target만 학습
            seqs.append(seq)
            labels.append(lab)
        L = max(s.size(0) for s in seqs)
        inp = torch.zeros(B, L, self.H, device=self.device, dtype=seqs[0].dtype)
        lab = torch.full((B, L), -100, dtype=torch.long, device=self.device)
        att = torch.zeros(B, L, dtype=torch.long, device=self.device)
        for b, (s, la) in enumerate(zip(seqs, labels)):                          # right-pad
            inp[b, :s.size(0)] = s
            lab[b, :la.size(0)] = la
            att[b, :s.size(0)] = 1
        return self.llm(inputs_embeds=inp, attention_mask=att, labels=lab).loss

    @torch.no_grad()
    def generate(self, region_masks, pre_ids: List[torch.Tensor], post_ids: PostIds,
                 region: str = "off", max_new_tokens: int = 1024,
                 cached=None, brain_ct=None) -> List[str]:
        """배치 생성(좌측 패딩) → 리포트 텍스트 리스트."""
        B = len(pre_ids)
        vis = self.encode_vision(region_masks, self._use_region(B, region),
                                 brain_ct=brain_ct, cached=cached)
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [i for i in {im_end, self.tok.eos_token_id} if i is not None]
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos_ids[0]

        seqs = []
        for b in range(B):
            post_e = self.embed(self._post(post_ids, b))
            seqs.append(torch.cat([self.embed(pre_ids[b]), vis[b].to(post_e.dtype), post_e], 0))
        maxL = max(s.size(0) for s in seqs)
        pad_emb = self.embed(torch.tensor([pad_id], device=self.device))[0]      # (H,)
        inp = pad_emb.view(1, 1, -1).expand(B, maxL, -1).contiguous()
        att = torch.zeros(B, maxL, dtype=torch.long, device=self.device)
        for b, s in enumerate(seqs):                                             # left-pad
            inp[b, maxL - s.size(0):] = s.to(inp.dtype)
            att[b, maxL - s.size(0):] = 1

        base_cfg = self.llm.config
        prev_attn = getattr(base_cfg, "_attn_implementation", None)
        switch = bool(self.cfg.gen_attn) and prev_attn is not None and prev_attn != self.cfg.gen_attn
        if switch:
            base_cfg._attn_implementation = self.cfg.gen_attn
        try:
            gen = self.llm.generate(
                inputs_embeds=inp, attention_mask=att,
                max_new_tokens=max_new_tokens, min_new_tokens=8,
                do_sample=False, num_beams=1,
                # greedy가 EOS를 못 뱉고 폭주하는 것만 억제. 더 센 제어는 문체를 흔들어 SAuto를 떨어뜨린다.
                repetition_penalty=1.1, no_repeat_ngram_size=6,
                eos_token_id=eos_ids, pad_token_id=pad_id, use_cache=True)
        finally:
            if switch:
                base_cfg._attn_implementation = prev_attn
        return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def load_trainable(model: PillarVLM, ckpt: str, device: str = "cuda") -> None:
    """stage1/stage2 체크포인트(학습 대상 텐서만) 로드. frozen 파트는 missing으로 뜨는 게 정상."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in {ckpt}: {list(unexpected)[:5]}"
    print(f"[load] {ckpt}: {len(sd)} tensors (missing {len(missing)}, unexpected 0)", flush=True)
