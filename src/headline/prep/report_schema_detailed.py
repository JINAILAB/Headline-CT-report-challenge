PRESENCE = {"type": "string", "enum": ["yes", "no", "uncertain"]}
LAT = {"type": "string", "enum": ["left", "right", "bilateral", "midline", "na"]}
NORMAL = {"type": "string", "enum": ["normal", "abnormal", "not_mentioned"]}

FINDING_CATEGORY = [
    "hemorrhage", "infarct", "edema", "mass", "mass_effect", "midline_shift", "herniation",
    "hydrocephalus", "ventricle_change", "hygroma", "extra_axial_collection", "pneumocephalus",
    "fracture", "atrophy", "small_vessel_disease", "calcification", "encephalomalacia_gliosis",
    "device", "postop_change", "sinus_mastoid", "orbit", "scalp_soft_tissue", "vascular", "other",
]
ACUITY = {"type": "string", "enum": ["hyperacute", "acute", "subacute", "chronic", "mixed", "na"]}
CHANGE = {"type": "string",
          "enum": ["new", "increased", "decreased", "resorbing", "unchanged", "resolved", "na"]}
CERTAINTY = {"type": "string",
             "enum": ["definite", "probable", "possible", "cannot_exclude", "na"]}

FINDING = {
    "type": "object", "additionalProperties": False,
    "required": ["category", "present", "laterality", "location", "subtype", "acuity",
                 "change", "certainty", "size", "differential", "descriptor"],
    "properties": {
        "category": {"type": "string", "enum": FINDING_CATEGORY},
        "present": PRESENCE,
        "laterality": LAT,
        "location": {"type": "string"},          # 구체적 위치(자유): "left frontobasal", "right petrous bone"
        "subtype": {"type": "string"},            # SDH/SAH/IPH/IVH/contusion 또는 자유("galeal hematoma","EVD")
        "acuity": ACUITY,
        "change": CHANGE,                          # prior 대비 변화
        "certainty": CERTAINTY,
        "size": {"type": "string"},               # "max 22 mm", "5 mm rightward" (없으면 "")
        "differential": {"type": "string"},       # 감별진단 텍스트(없으면 "")
        "descriptor": {"type": "string"},         # 원문 절 보존(손실 없는 재생성 앵커)
    },
}

DETAILED_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["study", "findings", "normal_survey", "impression"],
    "properties": {
        "study": {
            "type": "object", "additionalProperties": False,
            "required": ["prior_studies", "prior_modalities", "interval_change",
                         "interval_procedures", "limitations"],
            "properties": {
                "prior_studies": {"type": "string", "enum": ["none", "one", "multiple"]},
                "prior_modalities": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
                "interval_change": {"type": "string",
                                    "enum": ["none", "improved", "worsened", "mixed", "stable", "na"]},
                "interval_procedures": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
                "limitations": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
            },
        },
        "findings": {"type": "array", "maxItems": 40, "items": FINDING},
        "normal_survey": {
            "type": "object", "additionalProperties": False,
            "required": ["gray_white_differentiation", "basal_cisterns", "ventricles",
                         "midline_position", "extraaxial_csf_spaces", "sinuses", "mastoids",
                         "skull_bones", "orbits"],
            "properties": {k: NORMAL for k in
                           ["gray_white_differentiation", "basal_cisterns", "ventricles",
                            "midline_position", "extraaxial_csf_spaces", "sinuses", "mastoids",
                            "skull_bones", "orbits"]},
        },
        "impression": {
            "type": "object", "additionalProperties": False,
            "required": ["points", "recommendations"],
            "properties": {
                "points": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                "recommendations": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
            },
        },
    },
}

SYSTEM_PROMPT_DETAILED = (
    "You are a precise neuroradiology information-extraction engine. Decompose a non-contrast head CT "
    "report into a COMPLETE structured representation from which the original report could be fully "
    "regenerated. Output ONLY valid JSON matching the schema.\n"
    "PRINCIPLES\n"
    "- LOSSLESS: capture every clause. Create one 'findings' entry per distinct observation (each "
    "hemorrhage, infarct, device, fracture, sinus finding, etc. is its own entry). Do not merge or drop.\n"
    "- For each finding set 'descriptor' to the exact clause from the report (verbatim, trimmed) so the "
    "sentence can be reproduced; fill the structured fields (category, laterality, location, subtype, "
    "acuity, change, certainty, size, differential) from that clause.\n"
    "- Do NOT infer or invent anything not in the report. Use '' for a text field the report doesn't give, "
    "'na' for enums. Never hallucinate a fracture or finding.\n"
    "- NEGATION: an explicitly absent finding ('no hemorrhage', 'no infarct demarcation') => a findings "
    "entry with present='no' and the negated descriptor (needed to reproduce the 'No ...' sentence).\n"
    "- HEDGING/DD ('possible','cannot exclude','DD ...') => certainty accordingly and differential filled.\n"
    "- INFARCT acuity: 'demarcation'/'recent'/'increasingly hypodense' => acute/subacute; "
    "'old'/'encephalomalacia'/'lacunar'/'post-ischemic defect' => chronic.\n"
    "- FRACTURE present='yes' only if the report literally says fracture.\n"
    "- normal_survey: mark a region 'normal' only if the report states it is normal/unremarkable "
    "(e.g. 'preserved gray-white differentiation','basal cisterns clearly visible','normal ventricle "
    "width'); 'abnormal' if the report describes it as abnormal; else 'not_mentioned'.\n"
    "- study: prior comparison (none/one/multiple), prior modalities, overall interval_change, interim "
    "procedures/surgery, and assessment limitations (artifacts).\n"
    "- impression.points: the Assessment/Impression statements, prioritized; recommendations: any follow-up advice.\n"
    "Output only valid JSON."
)


def build_messages_detailed(report_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DETAILED},
        {"role": "user", "content": f"Report:\n{report_text}\n\nExtract the complete structured JSON."},
    ]


# 재생성(struct → 리포트) 프롬프트 — 재생성 검증용
RECONSTRUCT_SYSTEM = (
    "You are a neuroradiologist. You are given a STRUCTURED representation (JSON) of a non-contrast head "
    "CT report. Write the full radiology report in the original style: a '**Findings:**' section (prose, "
    "combining the findings and the stated-normal survey items, respecting laterality/location/size/change/"
    "negation/differential) followed by an '**Assessment:**' section from impression.points and "
    "recommendations. Use only information in the JSON. Do not add findings. Write natural radiology prose."
)
