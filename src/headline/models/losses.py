from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _aug_norm(lab: torch.Tensor) -> torch.Tensor:
    """multi-hot presence → is_normal 차원 추가 후 L2. 정상(all-0)끼리 코사인=1"""
    is_normal = (lab.sum(-1, keepdim=True) == 0).float()
    return F.normalize(torch.cat([lab, is_normal], dim=-1), dim=-1)


def siglip_global_loss(img: torch.Tensor, pos_txt: torch.Tensor,
                       logit_scale: torch.Tensor, logit_bias: torch.Tensor,
                       neg_txt: Optional[torch.Tensor] = None,
                       pos_lab: Optional[torch.Tensor] = None,
                       neg_lab: Optional[torch.Tensor] = None,
                       sim_mode: str = "label", mask_false_neg: bool = True,
                       tau_low: float = 0.55, tau_high: float = 0.90
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:

    B = img.size(0)
    cols = pos_txt if neg_txt is None else torch.cat([pos_txt, neg_txt], 0)      # (B+K,D)
    logits = logit_scale * (img @ cols.T) + logit_bias                            # (B,B+K)
    labels = -torch.ones_like(logits)
    idx = torch.arange(B, device=img.device)
    labels[idx, idx] = 1.0
    ignore = torch.zeros_like(logits, dtype=torch.bool)
    if mask_false_neg:
        with torch.no_grad():
            use_lab = sim_mode == "label" and pos_lab is not None
            if use_lab:
                col_lab = pos_lab if neg_lab is None else torch.cat([pos_lab, neg_lab], 0)
                S = _aug_norm(pos_lab) @ _aug_norm(col_lab).T                     # 라벨 코사인
            else:
                S = pos_txt @ cols.T                                              # 텍스트 코사인
            diag = torch.zeros_like(S, dtype=torch.bool)
            diag[idx, idx] = True
            labels = torch.where((S > tau_high) & ~diag, torch.ones_like(labels), labels)
            ignore = (S > tau_low) & (S <= tau_high) & ~diag
    ll = F.logsigmoid(labels * logits).masked_fill(ignore, 0.0)
    loss = -ll.sum() / (~ignore).sum().clamp(min=1)
    with torch.no_grad():
        stats = {"g_pos_logit": float(logits[idx, idx].mean()),
                 "g_r@1": float((logits.argmax(1) == idx).float().mean()),
                 "g_ignore_frac": float(ignore.float().mean())}
    return loss, stats


def local_sim_matrix(img_patch: torch.Tensor, txt_lines: torch.Tensor,
                     line_mask: torch.Tensor, tau_attn: float = 0.07) -> torch.Tensor:
    """GLoRIA region-attention → (B,B) 유사도.

    img_patch (B,P,D) L2, txt_lines (B,C,D) L2, line_mask (B,C) bool(유효 & groundable).
    (이미지 i, 리포트 j의 문장 c)마다: 문장으로 patch attention → attended vector v → cos(v, 문장),
    유효 문장에 대해 평균.
    """
    attn = torch.einsum("jcd,ipd->ijcp", txt_lines, img_patch) / tau_attn         # (Bi,Bj,C,P)
    a = attn.softmax(dim=-1)
    v = F.normalize(torch.einsum("ijcp,ipd->ijcd", a, img_patch), dim=-1)         # (Bi,Bj,C,D)
    cos = (v * txt_lines.unsqueeze(0)).sum(-1)                                    # (Bi,Bj,C)
    m = line_mask.unsqueeze(0).float()                                            # (1,Bj,C)
    return (cos * m).sum(-1) / m.sum(-1).clamp(min=1)                             # (Bi,Bj)


def local_infonce_loss(sim: torch.Tensor, logit_scale_local: torch.Tensor
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """local sim (B,B) → 대칭 InfoNCE (대각이 positive)."""
    B = sim.size(0)
    idx = torch.arange(B, device=sim.device)
    logits = logit_scale_local * sim
    loss = 0.5 * (F.cross_entropy(logits, idx) + F.cross_entropy(logits.T, idx))
    with torch.no_grad():
        stats = {"l_r@1": float((logits.argmax(1) == idx).float().mean()),
                 "l_pos_sim": float(sim[idx, idx].mean())}
    return loss, stats
