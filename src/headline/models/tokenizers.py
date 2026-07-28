from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pillar import FEAT_DIM, GLOBAL_DIM, SCALE_GRIDS


class Resampler(nn.Module):
    """num_queries개 latent가 시각 토큰(key/value)에 cross-attend → (B, num_queries, out_dim)."""

    def __init__(self, out_dim: int, num_queries: int = 48, n_heads: int = 8,
                 kv_dim: int = FEAT_DIM, depth: int = 2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, out_dim) * 0.02)
        self.kv_proj = nn.Linear(kv_dim, out_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_q": nn.LayerNorm(out_dim), "ln_kv": nn.LayerNorm(out_dim),
                "attn": nn.MultiheadAttention(out_dim, n_heads, batch_first=True),
                "ln_ff": nn.LayerNorm(out_dim),
                "ff": nn.Sequential(nn.Linear(out_dim, out_dim * 4), nn.GELU(),
                                    nn.Linear(out_dim * 4, out_dim)),
            }) for _ in range(depth)
        ])

    def forward(self, vis_tokens: torch.Tensor) -> torch.Tensor:
        # vis_tokens: (B, L, kv_dim)
        kv = self.kv_proj(vis_tokens)                       # (B,L,out)
        q = self.query.unsqueeze(0).expand(vis_tokens.size(0), -1, -1)   # (B,Q,out)
        for lyr in self.layers:
            a, _ = lyr["attn"](lyr["ln_q"](q), lyr["ln_kv"](kv), kv, need_weights=False)
            q = q + a
            q = q + lyr["ff"](lyr["ln_ff"](q))
        return q                                            # (B,Q,out)


class GlobalTokenizer(nn.Module):
    """multi-scale 그리드 + global_embed → global visual token 열 (B, Q+1, out_dim)."""

    def __init__(self, out_dim: int, num_queries: int = 48):
        super().__init__()
        self.resampler = Resampler(out_dim, num_queries=num_queries, kv_dim=FEAT_DIM)
        self.embed_proj = nn.Sequential(nn.LayerNorm(GLOBAL_DIM), nn.Linear(GLOBAL_DIM, out_dim))

    def forward(self, grids: List[torch.Tensor], global_embed: torch.Tensor) -> torch.Tensor:
        toks = [g.reshape(g.size(0), -1, g.size(-1)) for g in grids]   # [(B,DHW,384)]
        vis = torch.cat(toks, dim=1)                                   # (B, ΣDHW, 384)
        g_tok = self.resampler(vis)                                    # (B,Q,out)
        e_tok = self.embed_proj(global_embed).unsqueeze(1)             # (B,1,out)
        return torch.cat([e_tok, g_tok], dim=1)                        # (B,Q+1,out)


def masked_pool_grid(grid: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
 
    B, D, H, W, C = grid.shape
    V, R = D * H * W, masks.size(1)
    f = grid.reshape(B, V, C)                                    # (B,V,C)
    m = masks.reshape(B, R, V).to(f.dtype)                       # (B,R,V)
    num = torch.bmm(m, f)                                        # (B,R,C)
    den = m.sum(-1, keepdim=True).clamp_min(eps)                 # (B,R,1) — 빈 마스크도 NaN 없음
    return num / den


class RegionTokenizer(nn.Module):
    """[(B,D,H,W,384)×3] + region_masks (B,R,Dm,Hm,Wm) → (B,R,out_dim)."""

    def __init__(self, out_dim: int, in_dim: int = GLOBAL_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, grids: List[torch.Tensor], region_masks: torch.Tensor) -> torch.Tensor:
        assert region_masks.dim() == 5, region_masks.shape
        per_scale = []
        for grid, (D, H, W) in zip(grids, SCALE_GRIDS):
            m = F.interpolate(region_masks.float(), size=(D, H, W),
                              mode="trilinear", align_corners=False)   # (B,R,D,H,W)
            per_scale.append(masked_pool_grid(grid, m))                # (B,R,384)
        tok = torch.cat(per_scale, dim=-1)                             # (B,R,1152)
        return self.proj(tok)                                          # (B,R,out_dim)
