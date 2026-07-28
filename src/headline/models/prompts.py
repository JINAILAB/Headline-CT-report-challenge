from __future__ import annotations

import re
from typing import List, Optional, Tuple

SYSTEM = (
    "You are an experienced neuroradiologist reporting non-contrast head CT scans for a German "
    "university hospital. Follow the user's instruction exactly, using concise, precise radiological "
    "terminology and the department's reporting style."
)

GEN_INSTRUCTION = "Write the radiology report with a **Findings:** section and an **Assessment:** section."

# stage 1(visual tokenizer만 학습)은 원래 실행에서 아래의 단순 프롬프트를 씀.
# stage 2와 평가는 위의 SYSTEM/GEN_INSTRUCTION을 쓴다 — 재현을 위해 그 차이를 유지.
STAGE1_SYSTEM = ("You are a neuroradiologist. Given a non-contrast head CT, write the radiology "
                 "report with a Findings section and an Impression/Assessment section.")
STAGE1_INSTRUCTION = "Generate the radiology report."


def wrap_pre(user_context: str = "", system: str = SYSTEM) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_context}"


def wrap_post(instruction: str = GEN_INSTRUCTION) -> str:
    return f"{instruction}<|im_end|>\n<|im_start|>assistant\n"


def wrap_target(answer: str) -> str:
    return f"{answer}<|im_end|>"


# ── 리포트 텍스트 유틸 ────────────────────────────────────────────────────────
_FIND_RE = re.compile(r"\*{0,2}\s*findings?\s*:?\*{0,2}", re.I)
_ASSESS_RE = re.compile(r"\*{0,2}\s*(assessment|impression|interpretation|conclusion)\s*:?\*{0,2}", re.I)


def split_report(report: str) -> Tuple[Optional[str], Optional[str]]:
    """→ (findings 절, assessment 절). assessment 표지가 없으면 (None, None)."""
    m = _ASSESS_RE.search(report or "")
    if not m:
        return None, None
    findings = report[:m.start()]
    fm = _FIND_RE.search(findings)
    if fm:
        findings = findings[fm.end():]
    return findings.strip(" :\n"), report[m.end():].strip(" :\n")


def chunk_findings(find_text: str, max_chunks: int = 16) -> List[str]:

    parts = []
    for line in (find_text or "").split("\n"):
        line = line.strip(" -•*\t")
        if not line:
            continue
        for s in re.split(r"(?<=[.;])\s+", line):
            s = s.strip()
            if len(s) >= 8:
                parts.append(s)
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max_chunks:
            break
    return out
