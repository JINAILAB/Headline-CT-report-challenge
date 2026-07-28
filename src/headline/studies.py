from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

from . import paths


@dataclass(frozen=True)
class Study:
    """스터디 1건 (= 리포트 1건)."""

    split: str
    patient_id: str
    study_id: str
    dir: Path

    def recon_path(self, recon: str = "WT") -> Optional[Path]:
        """해당 recon(WT/KF)의 NIfTI 경로. 없으면 None.

        recon 폴더 이름이 토큰으로 시작하는 .nii.gz를 rglob으로 찾는다(중간 폴더 구조에 견고).
        """
        cands = [p for p in self.dir.rglob("*.nii.gz") if p.parent.name.startswith(recon)]
        return sorted(cands)[0] if cands else None

    @property
    def mask_dir(self) -> Path:
        return paths.mask_dir(self.split, self.patient_id, self.study_id)


def iter_studies(split: str, raw_root: Optional[Path] = None) -> Iterator[Study]:
    """data/raw/<split>/CT_data 아래 모든 스터디를 (patient, study) 정렬 순서로 순회."""
    root = (raw_root or paths.RAW) / split / "CT_data"
    if not root.is_dir():
        return
    for pat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for study_dir in sorted(s for s in pat_dir.iterdir() if s.is_dir()):
            yield Study(split, pat_dir.name, study_dir.name, study_dir)


def load_reports(split: str, raw_root: Optional[Path] = None) -> Dict[str, str]:
    """{study_id: report_text} — batch1/2/3 컬럼명 차이를 흡수한다."""
    sid_key, rep_key = paths.REPORT_CSV_COLS[split]
    csv_path = (raw_root / split / f"reports_train_{split}.csv") if raw_root else paths.report_csv(split)
    out: Dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row[sid_key]] = row[rep_key]
    return out


def load_all_reports(splits: Sequence[str] = paths.SPLITS,
                     raw_root: Optional[Path] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sp in splits:
        out.update(load_reports(sp, raw_root))
    return out


def study_patient_map(splits: Sequence[str] = paths.SPLITS,
                      raw_root: Optional[Path] = None) -> Dict[str, str]:
    """{study_id: patient_id} — 환자 단위 split에 필요."""
    return {st.study_id: st.patient_id
            for sp in splits for st in iter_studies(sp, raw_root)}
