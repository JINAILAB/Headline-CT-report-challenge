from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import paths

MASK_NAME = "hemorrhage_viola.nii.gz"
CSV_NAME = "hemorrhage_viola.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--viola_dir", type=Path, default=paths.REPO_ROOT / "third_party" / "violai",
                    help="Viola-Unet 체크아웃(main.py와 ckpt/가 있는 폴더)")
    ap.add_argument("--out_root", type=Path, default=paths.MASKS)
    ap.add_argument("--batch_size", type=int, default=20)
    a = ap.parse_args()

    from ..studies import iter_studies      # 늦게 import — viola 환경엔 torch만 있으면 된다

    studies = [s for i, s in enumerate(iter_studies(a.split)) if i % a.nshards == a.shard]
    todo = []
    for st in studies:
        out_dir = a.out_root / st.split / st.patient_id / st.study_id
        if (out_dir / MASK_NAME).exists() and (out_dir / MASK_NAME).stat().st_size > 0:
            continue                                  # 재개
        wt = st.recon_path("WT")
        if wt is not None:
            todo.append((st, wt, out_dir))
    print(f"[viola {a.shard}/{a.nshards}] split={a.split} shard={len(studies)} todo={len(todo)}",
          flush=True)
    if not todo:
        return

    stage = Path(tempfile.mkdtemp(prefix=f"viola_in_{a.split}_{a.shard}_"))
    predict = Path(tempfile.mkdtemp(prefix=f"viola_out_{a.split}_{a.shard}_"))
    id2out = {}
    for st, wt, out_dir in todo:
        link = stage / f"{st.study_id}.nii.gz"
        link.symlink_to(wt.resolve())
        id2out[st.study_id] = out_dir

    # cwd=viola_dir — ckpt 경로가 상대경로라 반드시 필요하다.
    rc = subprocess.run([sys.executable, "main.py", f"--input_dir={stage}",
                         f"--predict_dir={predict}", f"--batch_size={a.batch_size}"],
                        cwd=str(a.viola_dir)).returncode
    if rc != 0:
        sys.stderr.write(f"[viola {a.shard}] main.py rc={rc}\n")

    n_ok = 0
    out_dirs = sorted(predict.glob("predictions_violai_*"))
    if out_dirs:
        pdir = out_dirs[-1]
        rows = {}
        csv_file = pdir / "predictions_viola3-1.csv"
        if csv_file.exists():
            with open(csv_file, newline="") as fh:
                for r in csv.DictReader(fh):
                    rows[Path(r["File_Name"]).name] = r
        for sid, out_dir in id2out.items():
            src = pdir / f"{sid}.nii.gz"
            if not src.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(out_dir / MASK_NAME))
            row = rows.get(f"{sid}.nii.gz")
            if row:
                with open(out_dir / CSV_NAME, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    w.writeheader()
                    w.writerow(row)
            n_ok += 1
    print(f"[viola {a.shard}/{a.nshards}] 분배 {n_ok}/{len(todo)}", flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(predict, ignore_errors=True)


if __name__ == "__main__":
    main()
