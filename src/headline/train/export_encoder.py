from __future__ import annotations

import argparse
from pathlib import Path

from ..models.aligner import export_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = export_encoder(a.ckpt, a.out)
    print(f"[export] {n} tensors → {a.out}")


if __name__ == "__main__":
    main()
