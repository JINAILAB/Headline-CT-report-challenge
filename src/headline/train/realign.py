from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import paths
from ..data.contrastive_dataset import ContrastiveDataset, build_items, collate, load_presence_labels
from ..eval.core7 import collect_presence, core7_report, print_report, save_report
from ..labels import N_PRESENCE
from ..models.aligner import ContrastiveAligner
from ..models.losses import local_infonce_loss, local_sim_matrix, siglip_global_loss
from ..splits import load_val_patients


@dataclass
class RealignConfig:
    out: str = "outputs/aligner"
    epochs: int = 5
    bs: int = 16
    accum: int = 2
    enc_lr: float = 2e-5           # encoder는 낮게 — 사전학습 표현을 크게 흔들지 않는다
    head_lr: float = 3e-4          # aux head는 처음부터 학습
    warmup_frac: float = 0.05
    lam_local: float = 0.3         # 논문 lambda_loc
    lam_presence: float = 0.5      # 논문 lambda_pres
    sim_mode: str = "label"        # label(기본) | text — negative 정의의 근거
    mask_temporal: bool = True     # 비시각 문장을 local loss에서 제외
    neg_k: int = 512               # 스텝당 메모리 뱅크에서 뽑는 negative 수
    bank_max: int = 4096
    grad_clip: float = 5.0
    workers: int = 12
    seed: int = 42
    limit: int = 0                 # >0이면 스모크 모드
    splits: List[str] = field(default_factory=lambda: list(paths.SPLITS))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main(cfg: RealignConfig) -> None:
    _set_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_presence_labels(cfg.splits)
    items = build_items(cfg.splits, limit=cfg.limit, labels=labels)
    val_pats = load_val_patients()
    train_items = [it for it in items if it[3] not in val_pats]
    val_items = [it for it in items if it[3] in val_pats]
    if cfg.limit:                              # 스모크: 뒤쪽 일부를 val로 쓴다
        train_items, val_items = items[:-8] or items, items[-8:]
    print(f"[realign] train={len(train_items)} val={len(val_items)} "
          f"sim_mode={cfg.sim_mode} lam_local={cfg.lam_local} lam_presence={cfg.lam_presence}",
          flush=True)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(ContrastiveDataset(train_items, cfg.mask_temporal, cfg.splits, labels),
                          batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers,
                          collate_fn=collate, drop_last=True, generator=gen,
                          persistent_workers=cfg.workers > 0,
                          prefetch_factor=2 if cfg.workers > 0 else None)
    val_dl = DataLoader(ContrastiveDataset(val_items, cfg.mask_temporal, cfg.splits, labels),
                        batch_size=cfg.bs, shuffle=False, num_workers=min(4, cfg.workers),
                        collate_fn=collate)

    # 소견별 불균형이 커서(fracture 5%, orbit 0.5%) pos_weight를 준다. 20배에서 자른다.
    all_labels = torch.stack(list(labels.values()))
    p = all_labels.mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - p) / p).clamp(1.0, 20.0).cuda()

    model = ContrastiveAligner(n_presence=N_PRESENCE)
    enc_p, head_p = model.trainable_params()
    print(f"[realign] trainable enc={sum(q.numel() for q in enc_p)/1e6:.1f}M "
          f"head={sum(q.numel() for q in head_p)/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(
        [{"params": enc_p, "lr": cfg.enc_lr, "weight_decay": 0.0, "name": "enc"},
         {"params": head_p, "lr": cfg.head_lr, "weight_decay": 0.05, "name": "head"}],
        betas=(0.9, 0.98), eps=1e-6)

    steps_per_epoch = math.ceil(len(train_dl) / cfg.accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(10, int(total_steps * cfg.warmup_frac))

    bank_txt = torch.zeros(0, 4096, device="cuda")     # 과거 배치의 raw Qwen 임베딩
    bank_lab = torch.zeros(0, N_PRESENCE, device="cuda")
    best_auroc, step, t0 = -1.0, 0, time.time()
    history = []

    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        micro, running = 0, []
        for pe, find, lines, mask, lab in train_dl:
            pe, find, lab = pe.cuda(), find.cuda(), lab.cuda()
            img_g, img_p = model.encode_image_cached(pe)
            pos_txt = model.text_feat(find)                                   # (B,1152)

            neg_txt = neg_lab = None
            if bank_txt.size(0) >= cfg.neg_k:
                sel = torch.randint(0, bank_txt.size(0), (cfg.neg_k,), device="cuda")
                with torch.no_grad():
                    neg_txt = model.text_feat(bank_txt[sel])                   # frozen 투영
                neg_lab = bank_lab[sel]

            scale, bias, scale_local = model.scales()
            g_loss, g_stats = siglip_global_loss(img_g, pos_txt, scale, bias, neg_txt,
                                                 pos_lab=lab, neg_lab=neg_lab,
                                                 sim_mode=cfg.sim_mode)
            B, C = lines.shape[:2]
            line_txt = model.text_feat(lines.reshape(B * C, 4096).cuda()).reshape(B, C, 1152)
            l_loss, l_stats = local_infonce_loss(
                local_sim_matrix(img_p, line_txt, mask.cuda()), scale_local)
            bce = F.binary_cross_entropy_with_logits(model.presence_logits(img_g), lab,
                                                     pos_weight=pos_weight)
            loss = g_loss + cfg.lam_local * l_loss + cfg.lam_presence * bce

            if not math.isfinite(loss.item()):        # 비유한 스텝은 버리고 누적도 리셋
                print(f"  [nan] step{step} → skip", flush=True)
                opt.zero_grad()
                micro = 0
                continue
            (loss / cfg.accum).backward()
            running.append(loss.item())
            micro += 1
            bank_txt = torch.cat([bank_txt, find.detach()])[-cfg.bank_max:]
            bank_lab = torch.cat([bank_lab, lab.detach()])[-cfg.bank_max:]

            if micro % cfg.accum == 0:
                sc = _lr_scale(step, warmup, total_steps)
                for group in opt.param_groups:
                    group["lr"] = (cfg.enc_lr if group["name"] == "enc" else cfg.head_lr) * sc
                torch.nn.utils.clip_grad_norm_(
                    [q for g in opt.param_groups for q in g["params"]], cfg.grad_clip)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"  ep{ep} step{step}/{total_steps} loss={np.mean(running[-40:]):.3f} "
                          f"g={g_loss.item():.2f} l={l_loss.item():.2f} bce={bce.item():.2f} "
                          f"g_r@1={g_stats['g_r@1']:.2f} l_r@1={l_stats['l_r@1']:.2f} "
                          f"ign={g_stats['g_ignore_frac']:.2f} lr={cfg.enc_lr*sc:.1e} "
                          f"{(time.time()-t0)/60:.1f}min", flush=True)

        rep = core7_report(*collect_presence(model, val_dl))
        rep["epoch"] = ep
        history.append({"epoch": ep, "macro_auroc": rep["macro_auroc"], "macro_f1": rep["macro_f1"]})
        print(f"[eval ep{ep}] CORE7 macro AUROC={rep['macro_auroc']:.4f} "
              f"F1={rep['macro_f1']:.4f}", flush=True)
        if rep["macro_auroc"] > best_auroc:
            best_auroc = rep["macro_auroc"]
            torch.save({"aligner": model.state_dict(), "epoch": ep,
                        "macro_auroc": best_auroc}, out_dir / "aligner_bestauroc.pt")
            save_report(rep, out_dir / "core7_best.json")
            print_report(rep)
            print(f"    ↑ best CORE7 AUROC={best_auroc:.4f} → aligner_bestauroc.pt", flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(
        {"config": asdict(cfg), "history": history, "best_macro_auroc": best_auroc}, indent=2))
    print(f"[realign] done. best CORE7 macro AUROC={best_auroc:.4f} → {out_dir}", flush=True)

if __name__ == "__main__":
    from ..cli import parse_config
    main(parse_config(RealignConfig))
