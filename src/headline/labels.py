from __future__ import annotations

from typing import List

CORE7: List[str] = [
    "hemorrhage", "infarct", "fracture", "mass", "mass_effect", "midline_shift", "hydrocephalus",
]

PRESENCE_KEYS: List[str] = CORE7 + [
    "atrophy", "small_vessel_disease", "extra_axial_collection", "pneumocephalus", "edema",
    "herniation", "device", "postop_change", "sinus_mastoid", "calcification",
    "scalp_soft_tissue", "vascular", "orbit",
]

# 9개 "정상/비정상" 축 — detailed 스키마의 normal_survey에서 온다(0 미언급 / 1 정상 / 2 비정상).
SURVEY_KEYS: List[str] = [
    "gray_white_differentiation", "basal_cisterns", "ventricles", "midline_position",
    "extraaxial_csf_spaces", "sinuses", "mastoids", "skull_bones", "orbits",
]

N_PRESENCE = len(PRESENCE_KEYS)
assert PRESENCE_KEYS[:7] == CORE7
