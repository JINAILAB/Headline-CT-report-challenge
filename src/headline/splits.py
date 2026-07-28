from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, TypeVar

from . import paths

N_VAL_STUDIES = 600
SPLIT_SEED = 42

T = TypeVar("T")


def derive_val_patients(patient_of_study: Dict[str, str],
                        n_val: int = N_VAL_STUDIES,
                        seed: int = SPLIT_SEED) -> List[str]:
    """{study_id: patient_id} → val 환자 ID 리스트(정렬).

    환자 ID를 정렬해 결정적 기준 순서를 만든 뒤 seed 고정 셔플하고, 목표 스터디 수를 채울 때까지
    환자를 **통째로** val에 담는다. 마지막 환자 때문에 n_val을 약간 넘길 수 있다(스터디는 못 쪼갬).
    item 순서와 무관하게 환자 집합만으로 결정되므로 데이터셋 순회 순서가 달라도 동일하다.
    """
    by_pat: Dict[str, List[str]] = {}
    for sid, pid in patient_of_study.items():
        by_pat.setdefault(pid, []).append(sid)
    pats = sorted(by_pat)
    random.Random(seed).shuffle(pats)
    val_pats: List[str] = []
    n = 0
    for p in pats:
        if n >= n_val:
            break
        val_pats.append(p)
        n += len(by_pat[p])
    return sorted(val_pats)


def load_val_patients(path: Optional[Path] = None) -> Set[str]:
    """저장소에 고정된 val 환자 집합. 없으면 명확히 실패시킨다(조용한 fallback 금지).

    예전 코드는 파일이 없으면 md5 해시 기반의 *다른* split으로 조용히 넘어가, aligner와 VLM이
    서로 다른 val을 쓰는 누출을 만들었다. 여기서는 파일을 필수로 둔다.
    """
    p = path or paths.VAL_PATIENTS_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 없음. `uv run python -m headline.splits`로 생성하라 "
            "(원본 데이터가 있어야 한다). 이 파일이 aligner와 VLM의 val을 일치시킨다.")
    return set(json.load(open(p)))


def patient_holdout(items: Sequence[T], patient_of: Callable[[T], str],
                    val: bool, val_patients: Optional[Set[str]] = None) -> List[T]:
    """items를 val 환자 소속 여부로 나눈다. 원래 순서 보존."""
    vp = val_patients if val_patients is not None else load_val_patients()
    return [it for it in items if (patient_of(it) in vp) == val]


def _main() -> None:
    ap = argparse.ArgumentParser(description="내부 val split(환자 리스트) 생성")
    ap.add_argument("--out", type=Path, default=paths.VAL_PATIENTS_JSON)
    ap.add_argument("--n_val", type=int, default=N_VAL_STUDIES)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--splits", nargs="+", default=list(paths.SPLITS))
    args = ap.parse_args()

    from .studies import iter_studies, load_reports

    # universe = 리포트가 있는 모든 스터디(= 캐시 대상과 동일)
    patient_of: Dict[str, str] = {}
    for sp in args.splits:
        reports = load_reports(sp)
        for st in iter_studies(sp):
            if st.study_id in reports:
                patient_of[st.study_id] = st.patient_id
    pats = derive_val_patients(patient_of, args.n_val, args.seed)
    n_studies = sum(1 for pid in patient_of.values() if pid in set(pats))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pats, indent=0))
    print(f"[splits] studies={len(patient_of)} → val 환자 {len(pats)}명 / {n_studies} 스터디 → {args.out}")


if __name__ == "__main__":
    _main()
