from __future__ import annotations

import json

import pytest

from headline import paths
from headline.splits import derive_val_patients, load_val_patients, patient_holdout


def _toy_universe(n_patients: int = 500, studies_per: int = 2):
    return {f"S{p:04d}_{k}": f"P{p:04d}"
            for p in range(n_patients) for k in range(studies_per)}


def test_derive_is_deterministic():
    uni = _toy_universe()
    assert derive_val_patients(uni, 600, 42) == derive_val_patients(uni, 600, 42)


def test_derive_is_order_independent():
    """환자 집합만 같으면 순회 순서가 달라도 같은 split이 나와야 한다."""
    uni = _toy_universe()
    shuffled = dict(reversed(list(uni.items())))
    assert derive_val_patients(uni, 600, 42) == derive_val_patients(shuffled, 600, 42)


def test_derive_reaches_target_size():
    uni = _toy_universe()
    val = set(derive_val_patients(uni, 600, 42))
    n_val_studies = sum(1 for pid in uni.values() if pid in val)
    assert n_val_studies >= 600
    assert n_val_studies < 600 + max(2, 4)      # 마지막 환자만큼만 초과


def test_holdout_is_patient_disjoint():
    uni = _toy_universe()
    val = set(derive_val_patients(uni, 600, 42))
    items = list(uni.items())                    # (study_id, patient_id)
    tr = patient_holdout(items, lambda it: it[1], val=False, val_patients=val)
    va = patient_holdout(items, lambda it: it[1], val=True, val_patients=val)
    assert len(tr) + len(va) == len(items)
    assert not ({p for _, p in tr} & {p for _, p in va})


def test_example_split_file_has_the_expected_shape():
    """저장소에는 형식 예시만 들어 있다: 환자 ID 문자열의 JSON 리스트."""
    example = paths.SPLITS_DIR / "vlm_val_patients.example.json"
    pats = json.loads(example.read_text())
    assert isinstance(pats, list) and pats
    assert all(isinstance(p, str) and p for p in pats)


def test_split_file_is_required_when_missing():
    """split 파일이 없으면 조용한 fallback 대신 명확히 실패해야 한다."""
    if paths.VAL_PATIENTS_JSON.exists():
        assert load_val_patients() == set(json.loads(paths.VAL_PATIENTS_JSON.read_text()))
        return
    with pytest.raises(FileNotFoundError):
        load_val_patients()
