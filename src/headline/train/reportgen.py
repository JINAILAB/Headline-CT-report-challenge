from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import paths
from ..data.report_dataset import (AugmentedReportDataset, CachedReportDataset, make_collate,
                                   make_eval_collate)
from ..data.samplers import LengthGroupedBatchSampler
from ..eval.sauto import compute_sauto
from ..models.vlm import PillarVLM, VLMConfig, load_trainable

VISUAL_TOKEN_OFFSET = 70      # 길이 추정에 더할 시각 토큰 수(49 global + 18 region + 여유)


@dataclass
class ReportGenConfig:
    stage: int = 1
    out: str = "outputs/stage1"
    cache_root: str = ""                      # 비면 paths.PILLAR_CACHE
    qwen_path: str = ""                       # 비면 paths.QWEN_LLM
    init_from: Optional[str] = None           # stage 2가 stage 1 체크포인트를 warm-start
    epochs: int = 2
    lr: float = 1e-3                          # region/query/embed_proj (주 LR)
    lr_global: float = 0.0                    # 0이면 stage1=lr*0.4, stage2=lr
    accum_steps: int = 8
    warmup_ratio: float = 0.05
    lr_floor: float = 0.05                    # cosine 하한 — 후반 LR 고갈 방지
    grad_clip: float = 1.0
    region_dropout: float = 0.5
    # length-grouped 배칭 (토큰 예산으로 메모리 상한을 건다)
    max_tokens: int = 3072
    max_bs: int = 8
    megabatch_size: int = 256
    grad_ckpt: bool = False
    # 평가
    eval_every: int = 0                       # >0이면 N opt-step마다, 0이면 epoch 끝에만
    val_gen_n: int = 0                        # 생성 평가할 val 스터디 수(0=끔). stage2는 600.
    val_loss_batches: int = 128
    val_max_new: int = 1024
    gen_bs: int = 16
    # 기타
    workers: int = 4
    seed: int = 42
    limit: int = 0                            # >0이면 스모크 모드
    nan_patience: int = 100
    wandb: bool = False
    wandb_project: str = "headline-reportgen"
    wandb_name: Optional[str] = None
    splits: List[str] = field(default_factory=lambda: list(paths.SPLITS))

    def __post_init__(self):
        assert self.stage in (1, 2), self.stage
        if not self.cache_root:
            self.cache_root = str(paths.PILLAR_CACHE)
        if self.lr_global == 0.0:
            self.lr_global = self.lr * 0.4 if self.stage == 1 else self.lr


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: PillarVLM, val_dl: DataLoader, refs: List[str], cfg: ReportGenConfig) -> dict:
    """held-out val: teacher-forcing loss + 생성 S_Auto."""
    model.eval()
    total, nb = 0.0, 0
    for j, b in enumerate(val_dl):
        if cfg.val_loss_batches and j >= cfg.val_loss_batches:
            break
        total += model(b["region_masks"], b["pre_ids"], b["post_ids"], b["report_ids"],
                       region="off", cached=b["cached"]).item()
        nb += 1
    out = {"val_loss": total / max(1, nb)}
    if cfg.val_gen_n:
        gen_dl = DataLoader(val_dl.dataset, batch_size=cfg.gen_bs, shuffle=False,
                            num_workers=2, collate_fn=val_dl.collate_fn)
        preds: List[str] = []
        for b in gen_dl:
            if len(preds) >= cfg.val_gen_n:
                break
            preds.extend(model.generate(b["region_masks"], b["pre_ids"], b["post_ids"],
                                        region="off", max_new_tokens=cfg.val_max_new,
                                        cached=b["cached"]))
        preds = preds[:cfg.val_gen_n]
        out.update({f"val_{k}": v for k, v in compute_sauto(preds, refs[:len(preds)]).items()
                    if not k.endswith("_err")})
    model.train()
    return out


def build_dataloaders(cfg: ReportGenConfig, tokenizer):
    cache_root = Path(cfg.cache_root)
    if cfg.stage == 2:
        train_ds = AugmentedReportDataset(cache_root, cfg.splits, limit=cfg.limit)
        length_src = train_ds.length_texts()
        from collections import Counter
        print(f"[augment] tasks: {dict(Counter(train_ds.task_types()))}", flush=True)
    else:
        train_ds = CachedReportDataset(cache_root, cfg.splits, val=False, limit=cfg.limit)
        length_src = train_ds.report_texts()

    lens = [len(x) + VISUAL_TOKEN_OFFSET for x in tokenizer(length_src).input_ids]
    over = sum(l > cfg.max_tokens for l in lens)
    print(f"[batching] tokens max={max(lens)} mean={sum(lens)/len(lens):.0f} "
          f"| budget={cfg.max_tokens} 초과(bs=1 강제)={over}", flush=True)
    gen = torch.Generator().manual_seed(cfg.seed)
    sampler = LengthGroupedBatchSampler(lens, cfg.max_tokens, cfg.max_bs, cfg.megabatch_size, gen)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.workers,
                          collate_fn=make_collate(tokenizer))

    val_dl = val_refs = None
    if cfg.val_gen_n or cfg.val_loss_batches:
        val_ds = CachedReportDataset(cache_root, cfg.splits, val=True, limit=cfg.limit)
        # 평가 프롬프트는 데이터셋과 무관하게 고정한다(보고된 수치가 이 프롬프트로 나왔다).
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                            collate_fn=make_eval_collate(tokenizer))
        val_refs = val_ds.report_texts()
        print(f"[val] {len(val_ds)} studies (loss≤{cfg.val_loss_batches} batch, "
              f"gen={cfg.val_gen_n})", flush=True)
    return train_ds, train_dl, val_dl, val_refs


def build_optimizer(model: PillarVLM, cfg: ReportGenConfig):
    """큰 Perceiver 본체만 낮은 LR, 나머지(region/query/embed_proj/LoRA)는 주 LR."""
    low, high = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("global_tok") and ".resampler." in n and not n.endswith("resampler.query"):
            low.append(p)
        else:
            high.append(p)
    print(f"[lr] global_resampler={cfg.lr_global} ({sum(p.numel() for p in low)/1e6:.0f}M) "
          f"| rest={cfg.lr} ({sum(p.numel() for p in high)/1e6:.0f}M)", flush=True)
    return torch.optim.AdamW([{"params": low, "lr": cfg.lr_global},
                              {"params": high, "lr": cfg.lr}], weight_decay=0.0)


def main(cfg: ReportGenConfig) -> None:
    _set_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PillarVLM(VLMConfig(qwen_path=cfg.qwen_path, train_llm=(cfg.stage == 2),
                                region_dropout_p=cfg.region_dropout, grad_ckpt=cfg.grad_ckpt))
    if cfg.init_from:
        load_trainable(model, cfg.init_from)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[stage {cfg.stage}] trainable {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)

    train_ds, train_dl, val_dl, val_refs = build_dataloaders(cfg, model.tok)
    opt = build_optimizer(model, cfg)

    accum = max(1, cfg.accum_steps)
    steps_per_epoch = math.ceil(len(train_dl) / accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(30 if accum > 1 else 1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"[sched] dataset={len(train_ds)} batches/ep={len(train_dl)} "
          f"(avg bs≈{len(train_ds)/max(1,len(train_dl)):.1f}) accum={accum} "
          f"→ 유효배치≈{accum*len(train_ds)/max(1,len(train_dl)):.0f} | "
          f"warmup {warmup}/{total_steps}", flush=True)

    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_name or f"stage{cfg.stage}",
                         config=asdict(cfg))

    best_sauto, step, history = float("-inf"), 0, []

    def do_eval(tag: str) -> None:
        nonlocal best_sauto
        if val_dl is None:
            return
        torch.cuda.empty_cache()            # 생성 + BERTScore가 얹히므로 여유 확보
        ev = evaluate(model, val_dl, val_refs, cfg)
        torch.cuda.empty_cache()
        history.append({"tag": tag, "step": step, **ev})
        print("  [eval %s] " % tag + " ".join(f"{k}={v:.4f}" for k, v in ev.items()), flush=True)
        if run is not None:
            run.log(ev, step=step)
        sa = ev.get("val_sauto")
        if sa is not None and sa == sa and sa > best_sauto:
            best_sauto = sa
            torch.save(model.trainable_state_dict(), out_dir / f"ckpt_stage{cfg.stage}_best.pt")
            print(f"    ↑ best val_sauto={sa:.4f} → ckpt_stage{cfg.stage}_best.pt", flush=True)

    model.train()
    losses, step_times, nan_run, micro, good, window = [], [], 0, 0, 0, []
    opt.zero_grad(set_to_none=True)
    t_prev, done = time.time(), False

    for ep in range(cfg.epochs):
        for batch in train_dl:
            loss = model(batch["region_masks"], batch["pre_ids"], batch["post_ids"],
                         batch["report_ids"], region="dropout", cached=batch["cached"])
            lval = loss.item()
            if math.isfinite(lval):
                (loss / accum).backward()
                good += 1
                window.append(lval)
            else:                                   # 비유한 micro는 backward 생략(누적 grad 보존)
                nan_run += 1
                if cfg.nan_patience and nan_run >= cfg.nan_patience:
                    print(f"[abort] 연속 {nan_run} non-finite → 중단", flush=True)
                    done = True
            micro += 1

            if micro >= accum or done:
                if good:
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                    if torch.isfinite(gnorm):
                        nan_run = 0
                        opt.step()
                        sched.step()
                        step += 1
                        torch.cuda.synchronize()
                        dt, t_prev = time.time() - t_prev, time.time()
                        step_times.append(dt)
                        losses.append(sum(window) / len(window))
                        if run is not None:
                            run.log({"loss": losses[-1], "lr": sched.get_last_lr()[-1],
                                     "grad_norm": float(gnorm), "sec_per_step": dt, "epoch": ep},
                                    step=step)
                        if step % 5 == 0 or step == 1:
                            print(f"  step {step}/{total_steps}: loss={losses[-1]:.4f} "
                                  f"(avg5={np.mean(losses[-5:]):.4f}) gnorm={float(gnorm):.2f} "
                                  f"lr={sched.get_last_lr()[-1]:.2e} {dt:.2f}s/it", flush=True)
                        if cfg.eval_every and step % cfg.eval_every == 0:
                            do_eval(f"step{step}")
                    else:                            # 누적 grad가 비유한 → 이 업데이트 폐기
                        nan_run += 1
                        print(f"  step {step}: non-finite grad → skip (연속 {nan_run})", flush=True)
                opt.zero_grad(set_to_none=True)
                micro, good, window = 0, 0, []
            if done:
                break
        do_eval(f"ep{ep}")
        if done:
            break

    if not losses:
        print("[abort] 정상 step 0건 — 체크포인트를 저장하지 않는다.", flush=True)
        return

    warm = step_times[2:] or step_times
    med = sorted(warm)[len(warm) // 2]
    print(f"[timing] median {med:.2f}s/step | 1 epoch({len(train_dl)} batch) "
          f"≈ {med*len(train_dl)/accum/60:.0f}분", flush=True)

    sd = model.trainable_state_dict()
    torch.save(sd, out_dir / f"ckpt_stage{cfg.stage}.pt")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"config": asdict(cfg), "history": history, "best_val_sauto": best_sauto,
         "first_loss": losses[0], "last_loss": losses[-1]}, indent=2))
    print(f"[done] {out_dir}/ckpt_stage{cfg.stage}.pt ({len(sd)} tensors) "
          f"| best_val_sauto={best_sauto:.4f}", flush=True)
    if run is not None:
        run.finish()

if __name__ == "__main__":
    from ..cli import parse_config
    main(parse_config(ReportGenConfig))
