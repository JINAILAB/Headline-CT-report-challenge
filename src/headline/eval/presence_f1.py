from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import paths
from ..labels import CORE7


def _present(entry) -> int:
    if isinstance(entry, dict):
        return int(str(entry.get("present", "no")).lower() == "yes")
    return int(str(entry).lower() == "yes")


def _presence_vector(doc: dict) -> Dict[str, int]:
    return {f: _present(doc.get(f)) for f in CORE7}


def _find_gold(gold_root: Path, sid: str, splits) -> Optional[Path]:
    for sp in splits:
        p = gold_root / sp / f"{sid}.json"
        if p.exists():
            return p
    p = gold_root / f"{sid}.json"                  # split 하위폴더가 없는 배치도 허용
    return p if p.exists() else None


def macro_f1(pairs: List[Tuple[Dict[str, int], Dict[str, int]]]) -> dict:
    per = {}
    for f in CORE7:
        tp = sum(g[f] == 1 and p[f] == 1 for g, p in pairs)
        fp = sum(g[f] == 0 and p[f] == 1 for g, p in pairs)
        fn = sum(g[f] == 1 and p[f] == 0 for g, p in pairs)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[f] = {"f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                  "precision": prec, "recall": rec, "n_pos": tp + fn}
    tp = sum(g[f] == 1 and p[f] == 1 for g, p in pairs for f in CORE7)
    fp = sum(g[f] == 0 and p[f] == 1 for g, p in pairs for f in CORE7)
    fn = sum(g[f] == 1 and p[f] == 0 for g, p in pairs for f in CORE7)
    mprec = tp / (tp + fp) if (tp + fp) else 0.0
    mrec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"per_finding": per,
            "macro_f1": sum(v["f1"] for v in per.values()) / len(CORE7),
            "micro_f1": 2 * mprec * mrec / (mprec + mrec) if (mprec + mrec) else 0.0,
            "n": len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path, help="생성 리포트 추출 결과 디렉터리")
    ap.add_argument("--gold_root", type=Path, default=paths.REPORT_LABELS / "gptoss",
                    help="참조 리포트 추출 결과 루트(<split>/<sid>.json)")
    ap.add_argument("--out", type=Path, default=None, help="결과 JSON 경로")
    a = ap.parse_args()

    pairs, missing = [], 0
    for pf in sorted(a.pred_dir.glob("*.json")):
        gp = _find_gold(a.gold_root, pf.stem, paths.SPLITS)
        if gp is None:
            missing += 1
            continue
        pairs.append((_presence_vector(json.loads(gp.read_text())),
                      _presence_vector(json.loads(pf.read_text()))))
    assert pairs, f"{a.pred_dir}와 {a.gold_root}가 겹치는 study가 없다"
    rep = macro_f1(pairs)

    print(f"\n=== CORE7 presence-F1 (n={rep['n']}, gold 없음={missing}) ===")
    print(f"  macro_f1 = {rep['macro_f1']:.4f}   micro_f1 = {rep['micro_f1']:.4f}")
    for f, v in rep["per_finding"].items():
        print(f"  {f:14s} f1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  "
              f"pos={v['n_pos']}")
    out = a.out or (a.pred_dir.parent / "presence_f1.json")
    out.write_text(json.dumps(rep, indent=2))
    print(f"  → {out}", flush=True)


if __name__ == "__main__":
    main()
