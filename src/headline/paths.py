from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/headline/paths.py → <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("HEADLINE_DATA_ROOT", REPO_ROOT / "data")).resolve()

RAW = DATA_ROOT / "raw"                 # raw/<split>/CT_data/<pid>/<sid>/<recon>/*.nii.gz
INTERIM = DATA_ROOT / "interim"
PROCESSED = DATA_ROOT / "processed"

MASKS = INTERIM / "masks"               # <split>/<pid>/<sid>/{brain_regions,bone_craniofacial,hemorrhage_viola}.nii.gz
REPORT_LABELS = INTERIM / "report_labels"   # <extractor>/<split>/<sid>.json
TEXT_CACHE = INTERIM / "text_cache"     # <split>/<sid>.pt  (Qwen3-Embedding-8B)
PE_CACHE = INTERIM / "pe_cache"         # <split>/<sid>.pt  (frozen patch_embed 출력)
PILLAR_CACHE = INTERIM / "pillar_cache"  # <split>/<sid>.pt (grids + global_embed + region_masks32)

CLASSIFIER_LABELS = PROCESSED / "classifier_labels"   # <split>.jsonl (presence 20 + survey 9)
AUGMENT = PROCESSED / "augment"                       # <split>.jsonl (report-to-task SFT 인스턴스)

# 저장소에 포함되는 고정 산출물
SPLITS_DIR = REPO_ROOT / "splits"
VAL_PATIENTS_JSON = SPLITS_DIR / "vlm_val_patients.json"
CONFIGS = REPO_ROOT / "configs"

# 외부 가중치(다운로드 위치는 docs/DATA.md 참고). MODELS_ROOT로 재지정 가능.
MODELS_ROOT = Path(os.environ.get("HEADLINE_MODELS_ROOT", REPO_ROOT / "models")).resolve()
PILLAR0 = Path(os.environ.get("HEADLINE_PILLAR0", REPO_ROOT / "refs" / "pillar0_headct")).resolve()
QWEN_LLM = MODELS_ROOT / "qwen2.5-7b-instruct"        # report generator base LLM
QWEN_EMBED = MODELS_ROOT / "qwen3-embedding-8b"       # frozen text encoder (re-alignment)
GPT_OSS = MODELS_ROOT / "gpt-oss-120b"                # finding extractor (labels + eval)
MEDGEMMA = MODELS_ROOT / "medgemma-27b"               # 2nd extractor (agreement signal only)

SPLITS = ("batch1", "batch2", "batch3")

# batch1과 batch2/3의 리포트 CSV 컬럼명이 다르다 — 로더에서 통일한다.
REPORT_CSV_COLS = {
    "batch1": ("StudyID", "Report"),
    "batch2": ("pseudo_study_id", "report"),
    "batch3": ("pseudo_study_id", "report"),
}


def report_csv(split: str) -> Path:
    return RAW / split / f"reports_train_{split}.csv"


def mask_dir(split: str, patient_id: str, study_id: str) -> Path:
    return MASKS / split / patient_id / study_id
