from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .. import paths
from ..labels import CORE7
from ..models.prompts import GEN_INSTRUCTION, split_report
from ..studies import load_reports

# 스터디당 보조 태스크 추가 확률 (리포트:보조 ≈ 60:40)
AUX_P = {"impression_synthesis": 0.30, "structured_extraction": 0.12,
         "atomic_qa": 0.13, "nli_verification": 0.06, "report_correction": 0.06}

_OTHER_WORD = {"atrophy": "atrophy", "small_vessel_disease": "small-vessel disease",
               "extra_axial_collection": "extra-axial collection", "pneumocephalus": "pneumocephalus",
               "edema": "edema", "herniation": "herniation", "postop": "post-operative changes"}


def _attrs(d: dict, keys) -> List[str]:
    out = []
    for k in keys:
        v = d.get(k)
        if isinstance(v, list):
            v = list(dict.fromkeys(x for x in v if x and x != "na"))
            if v:
                out.append("/".join(v))
        elif v and v not in ("na", "", None):
            out.append(str(v))
    return out


def _line(name: str, d: dict, attr_keys=()) -> str:
    present = d.get("present", "no")
    if present == "no":
        return f"{name}: absent"
    if present == "uncertain":
        return f"{name}: questionable"
    parts = _attrs(d, attr_keys)
    return f"{name}: " + (", ".join(parts) if parts else "present")


def render_scaffold(s: dict) -> str:
    """구조화 소견 dict → 사람이 읽는 한 줄씩의 소견 목록 (structured_extraction의 정답).

    other.comparison(이전 검사 대비)은 단일 볼륨으로 알 수 없으므로 제외한다.
    """
    lines = [
        _line("Hemorrhage", s.get("hemorrhage", {}), ["acuity", "subtypes", "laterality", "locations"]),
        _line("Infarct", s.get("infarct", {}), ["acuity", "laterality", "locations"]),
    ]
    fr = s.get("fracture", {})
    if fr.get("present") == "yes":
        loc = fr.get("location")
        lines.append("Fracture: " + (loc if loc and loc != "na" else "present"))
    else:
        lines.append(_line("Fracture", fr))
    lines.append(_line("Mass", s.get("mass", {}), ["laterality", "locations"]))
    lines.append(_line("Mass effect", s.get("mass_effect", {})))
    mls = s.get("midline_shift", {})
    if mls.get("present") == "yes":
        seg = [f"{mls['mm']} mm"] if mls.get("mm") not in (None, "na") else []
        if mls.get("direction") not in ("na", None, ""):
            seg.append(str(mls["direction"]))
        lines.append("Midline shift: " + (" ".join(seg) if seg else "present"))
    else:
        lines.append(_line("Midline shift", mls))
    lines.append(_line("Hydrocephalus", s.get("hydrocephalus", {})))

    other = s.get("other", {})
    extra = [w for k, w in _OTHER_WORD.items() if other.get(k) == "yes"]
    devices = [d for d in (other.get("devices") or []) if d and d != "na"]
    if devices:
        extra.append("devices: " + ", ".join(devices))
    if extra:
        lines.append("Additional: " + "; ".join(extra))
    return "\n".join(lines)


def _qa(s: dict, rng: random.Random) -> Optional[Tuple[str, str]]:
    cands = []
    for c in CORE7:
        d = s.get(c, {})
        present = d.get("present", "no")
        name = c.replace("_", " ")
        cands.append((f"Is there {name}?",
                      {"yes": "Yes.", "no": "No.", "uncertain": "Possibly."}.get(present, "No.")))
        if present != "yes":
            continue
        if d.get("laterality") and d["laterality"] != "na":
            cands.append((f"What is the laterality of the {name}?", d["laterality"].capitalize() + "."))
        if c in ("hemorrhage", "infarct") and d.get("acuity") and d["acuity"] != "na":
            cands.append((f"What is the acuity of the {name}?", d["acuity"].capitalize() + "."))
        if c == "hemorrhage":
            subs = [x for x in (d.get("subtypes") or []) if x != "na"]
            if subs:
                cands.append(("What hemorrhage subtype is present?", "/".join(subs) + "."))
    return rng.choice(cands) if cands else None


def _nli(s: dict, rng: random.Random) -> Optional[Tuple[str, str]]:
    yes = [c for c in CORE7 if s.get(c, {}).get("present") == "yes"]
    no = [c for c in CORE7 if s.get(c, {}).get("present") == "no"]
    if yes and rng.random() < 0.55:
        c = rng.choice(yes)
        name = c.replace("_", " ")
        lat = s[c].get("laterality")
        if lat in ("left", "right") and rng.random() < 0.5:
            return f"There is {name} on the {'right' if lat == 'left' else 'left'}.", "Contradicted."
        return f"There is {name}.", "Supported."
    if no:
        return f"There is {rng.choice(no).replace('_', ' ')}.", "Contradicted."
    return None


def _correction(s: dict, rng: random.Random) -> Optional[Tuple[str, str]]:
    cands = [c for c in CORE7 if s.get(c, {}).get("present") == "yes"
             and s[c].get("laterality") in ("left", "right")]
    if not cands:
        return None
    c = rng.choice(cands)
    d = s[c]
    name, lat = c.replace("_", " "), d["laterality"]
    flip = "right" if lat == "left" else "left"
    locs = list(dict.fromkeys(x for x in (d.get("locations") or []) if x != "na"))
    tail = f" in the {'/'.join(locs)}" if locs else ""
    return f"There is {name} on the {flip}{tail}.", f"There is {name} on the {lat}{tail}."


def build_study(split: str, sid: str, report: str, s: dict, rng: random.Random) -> List[dict]:
    inst: List[dict] = []

    def add(task: str, instruction: str, target: str, user_context: str = "") -> None:
        if target and target.strip():
            inst.append({"study_id": sid, "split": split, "task_type": task,
                         "user_context": user_context, "instruction": instruction,
                         "target": target})

    add("report", GEN_INSTRUCTION, report)              # 항상 1건, 평가와 동일한 형식

    findings, assessment = split_report(report)
    if rng.random() < AUX_P["impression_synthesis"] and findings and assessment:
        add("impression_synthesis", "Write the **Assessment:** (impression) for this study.",
            assessment, user_context=f"Findings:\n{findings}\n")
    if rng.random() < AUX_P["structured_extraction"]:
        add("structured_extraction",
            "List the structured imaging findings of the current scan, one per line.",
            render_scaffold(s))
    if rng.random() < AUX_P["atomic_qa"]:
        qa = _qa(s, rng)
        if qa:
            add("atomic_qa", qa[0], qa[1])
    if rng.random() < AUX_P["nli_verification"]:
        nli = _nli(s, rng)
        if nli:
            add("nli_verification",
                f'Statement: "{nli[0]}" Based on the CT, is this supported, contradicted, '
                "or not established?", nli[1])
    if rng.random() < AUX_P["report_correction"]:
        cor = _correction(s, rng)
        if cor:
            add("report_correction",
                "Correct any statement in the draft that is inconsistent with the CT; "
                "output the corrected sentence.", cor[1], user_context=f"Draft: {cor[0]}\n")
    return inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_root", type=Path, default=paths.REPORT_LABELS / "gptoss")
    ap.add_argument("--out", type=Path, default=paths.AUGMENT)
    ap.add_argument("--splits", nargs="+", default=list(paths.SPLITS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    tasks = Counter()
    for split in a.splits:
        reports = load_reports(split)
        files = sorted((a.labels_root / split).glob("*.json"))
        if a.limit:
            files = files[:a.limit]
        n = 0
        with open(a.out / f"{split}.jsonl", "w") as fh:
            for f in files:
                if f.stem not in reports:
                    continue
                s = json.loads(f.read_text())
                if "_parse_error" in s:
                    continue
                for item in build_study(split, f.stem, reports[f.stem], s, rng):
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    tasks[item["task_type"]] += 1
                    n += 1
        print(f"  {split}: studies={len(files)} instances={n}", flush=True)

    total = sum(tasks.values())
    print(f"[augment] total={total}  task 분포={dict(tasks)}")
    print(f"  리포트 작성 비중 = {100*tasks['report']/max(total,1):.0f}%")


if __name__ == "__main__":
    main()
