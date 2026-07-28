from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .. import paths
from ..labels import CORE7, PRESENCE_KEYS, SURVEY_KEYS
from ..studies import iter_studies

# core 스키마의 other.* → presence 라벨명
OTHER_KEYS = ["atrophy", "small_vessel_disease", "extra_axial_collection",
              "pneumocephalus", "edema", "herniation"]
# detailed findings[].category → presence 라벨명 (양성일 때만 리포트에 등장하는 항목)
DETAILED_ONLY = ["calcification", "scalp_soft_tissue", "vascular"]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return {} if "_parse_error" in doc else doc


def project_one(core: dict, detailed: dict) -> dict:
    presence = {k: 0 for k in PRESENCE_KEYS}
    for k in CORE7:
        if core.get(k, {}).get("present") == "yes":
            presence[k] = 1
    other = core.get("other", {})
    for k in OTHER_KEYS:
        if other.get(k) == "yes":
            presence[k] = 1
    if other.get("devices"):
        presence["device"] = 1
    if other.get("postop") == "yes":
        presence["postop_change"] = 1

    survey_raw = detailed.get("normal_survey") or {}
    survey = {k: {"normal": 1, "abnormal": 2}.get(survey_raw.get(k), 0) for k in SURVEY_KEYS}
    presence["sinus_mastoid"] = int(survey.get("sinuses") == 2 or survey.get("mastoids") == 2)
    presence["orbit"] = int(survey.get("orbits") == 2)

    categories = collections.Counter(f.get("category") for f in (detailed.get("findings") or [])
                                     if f.get("present") == "yes")
    for c in DETAILED_ONLY:
        if categories.get(c):
            presence[c] = 1
    return {"presence": presence, "survey": survey}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_root", type=Path, default=paths.REPORT_LABELS / "gptoss",
                    help="core 스키마 추출 결과(<split>/<sid>.json)")
    ap.add_argument("--detailed_root", type=Path, default=paths.REPORT_LABELS / "gptoss_detailed")
    ap.add_argument("--out", type=Path, default=paths.CLASSIFIER_LABELS)
    ap.add_argument("--splits", nargs="+", default=list(paths.SPLITS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    total = collections.Counter()
    n_all = 0
    for sp in a.splits:
        pid_of = {st.study_id: st.patient_id for st in iter_studies(sp)}
        rows = []
        for cf in sorted((a.core_root / sp).glob("*.json")):
            core = _load(cf)
            if not core:
                continue
            rec = project_one(core, _load(a.detailed_root / sp / f"{cf.stem}.json"))
            rows.append({"study_id": cf.stem, "patient_id": pid_of.get(cf.stem, "NA"),
                         "split": sp, **rec})
            total.update({k for k, v in rec["presence"].items() if v})
        with open(a.out / f"{sp}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_all += len(rows)
        print(f"  {sp}: {len(rows)} studies → {a.out / f'{sp}.jsonl'}", flush=True)

    print(f"[project_labels] {n_all} studies")
    print("  양성 비율(%):", {k: round(100 * total[k] / max(n_all, 1), 1) for k in PRESENCE_KEYS})


if __name__ == "__main__":
    main()
