from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import paths
from ..data.report_dataset import CachedReportDataset, make_eval_collate
from ..models.vlm import PillarVLM, VLMConfig, load_trainable
from .sauto import compute_sauto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage 2 체크포인트(ckpt_stage2_best.pt)")
    ap.add_argument("--cache_root", default=str(paths.PILLAR_CACHE))
    ap.add_argument("--qwen_path", default="")
    ap.add_argument("--out", required=True, help="생성 리포트 CSV 경로")
    ap.add_argument("--metrics_out", default="", help="S_Auto JSON 경로(기본: <out>.sauto.json)")
    ap.add_argument("--gen_bs", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    model = PillarVLM(VLMConfig(qwen_path=a.qwen_path, train_llm=True))
    load_trainable(model, a.ckpt)
    model.eval()

    ds = CachedReportDataset(Path(a.cache_root), val=True, limit=a.limit)
    dl = DataLoader(ds, batch_size=a.gen_bs, shuffle=False, num_workers=2,
                    collate_fn=make_eval_collate(model.tok))
    sids, refs = ds.study_ids(), ds.report_texts()
    print(f"[gen] val={len(ds)} cache={a.cache_root}", flush=True)

    preds = []
    for i, b in enumerate(dl):
        preds.extend(model.generate(b["region_masks"], b["pre_ids"], b["post_ids"],
                                    region="off", max_new_tokens=a.max_new, cached=b["cached"]))
        if (i + 1) % 5 == 0:
            print(f"  {len(preds)}/{len(ds)}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study_id", "report", "reference"])
        for sid, pred, ref in zip(sids, preds, refs):
            w.writerow([sid, pred, ref])
    print(f"[gen] {len(preds)} reports → {out}", flush=True)

    del model
    torch.cuda.empty_cache()                    # BERTScore 로드 전에 GPU를 비운다
    metrics = compute_sauto(preds, refs[:len(preds)])
    mpath = Path(a.metrics_out) if a.metrics_out else out.with_suffix(".sauto.json")
    mpath.write_text(json.dumps({"n": len(preds), **metrics}, indent=2))
    print(f"\n=== S_Auto (n={len(preds)}) ===")
    print(f"  S_Auto     = {metrics['sauto']:.4f}")
    print(f"  BERTScore  = {metrics['bertscore_f1']:.4f}   METEOR = {metrics['meteor']:.4f}")
    print(f"  ROUGE-L    = {metrics['rouge_l']:.4f}   BLEU-4 = {metrics['bleu4']:.4f}")
    print(f"  → {mpath}", flush=True)


if __name__ == "__main__":
    main()
