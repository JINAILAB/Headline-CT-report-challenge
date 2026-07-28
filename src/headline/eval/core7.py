from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..labels import CORE7, PRESENCE_KEYS
from .metrics_util import auroc, best_f1


@torch.no_grad()
def collect_presence(model, dataloader) -> tuple:
    """(P, Y) — (N,20) 예측 확률과 정답."""
    model.eval()
    P, Y = [], []
    for pe, _find, _lines, _mask, lab in dataloader:
        img_g, _ = model.encode_image_cached(pe.cuda())
        P.append(torch.sigmoid(model.presence_logits(img_g)).float().cpu().numpy())
        Y.append(lab.numpy())
    model.train()
    return np.concatenate(P), np.concatenate(Y)


def core7_report(P: np.ndarray, Y: np.ndarray) -> Dict[str, object]:
    """per-finding AUROC/F1 + CORE7 macro."""
    per: Dict[str, dict] = {}
    for name in CORE7:
        j = PRESENCE_KEYS.index(name)
        y, p = Y[:, j], P[:, j]
        per[name] = {"auroc": auroc(y, p), "f1": best_f1(y, p), "n_pos": int(y.sum())}
    aucs = [v["auroc"] for v in per.values() if v["auroc"] is not None]
    f1s = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_finding": per,
            "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "n_studies": int(len(Y))}


def print_report(rep: Dict[str, object]) -> None:
    print(f"  val n={rep['n_studies']}")
    for name, v in rep["per_finding"].items():
        a = v["auroc"] if v["auroc"] is not None else float("nan")
        f = v["f1"] if v["f1"] is not None else float("nan")
        print(f"  {name:16s} AUROC {a:.3f}  F1 {f:.3f}  (n_pos={v['n_pos']})")
    print(f"  CORE7 macro AUROC = {rep['macro_auroc']:.4f} | macro F1 = {rep['macro_f1']:.4f}")


def save_report(rep: Dict[str, object], path: Optional[Path]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rep, indent=2))
