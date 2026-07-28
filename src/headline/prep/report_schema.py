PRESENCE = {"type": "string", "enum": ["yes", "no", "uncertain"]}
LATERALITY = {"type": "string", "enum": ["left", "right", "bilateral", "midline", "na"]}
BRAIN_LOC = ["frontal", "parietal", "temporal", "occipital", "insular", "cerebellum",
             "brainstem", "thalamus", "basal_ganglia", "internal_capsule", "ventricle",
             "periventricular", "corpus_callosum", "convexity", "tentorial", "na"]


def _loc_array():
    # maxItems: guided decoding이 배열을 강제 종료 → medgemma 퇴행성 반복(무한 배열) 방지
    return {"type": "array", "maxItems": 6, "items": {"type": "string", "enum": BRAIN_LOC}}


# guided_json 스키마 (vLLM). 각 소견: present + 속성. acuity로 만성/급성 구분(경색 FP 핵심).
REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hemorrhage", "infarct", "fracture", "mass", "mass_effect",
                 "midline_shift", "hydrocephalus", "other"],
    "properties": {
        "hemorrhage": {
            "type": "object", "additionalProperties": False,
            "required": ["present", "subtypes", "laterality", "locations", "acuity"],
            "properties": {
                "present": PRESENCE,
                "subtypes": {"type": "array", "maxItems": 6, "items": {"type": "string",
                              "enum": ["EDH", "SDH", "SAH", "IPH", "IVH", "contusion", "na"]}},
                "laterality": LATERALITY, "locations": _loc_array(),
                "acuity": {"type": "string", "enum": ["acute", "chronic", "mixed", "na"]},
            },
        },
        "infarct": {
            "type": "object", "additionalProperties": False,
            "required": ["present", "laterality", "locations", "acuity"],
            "properties": {
                "present": PRESENCE, "laterality": LATERALITY, "locations": _loc_array(),
                "acuity": {"type": "string",
                           "enum": ["acute", "subacute", "chronic", "lacunar", "na"]},
            },
        },
        "fracture": {
            "type": "object", "additionalProperties": False,
            "required": ["present", "location"],
            "properties": {"present": PRESENCE, "location": {"type": "string"}},
        },
        "mass": {
            "type": "object", "additionalProperties": False,
            "required": ["present", "laterality", "locations"],
            "properties": {"present": PRESENCE, "laterality": LATERALITY, "locations": _loc_array()},
        },
        "mass_effect": {
            "type": "object", "additionalProperties": False,
            "required": ["present"], "properties": {"present": PRESENCE},
        },
        "midline_shift": {
            "type": "object", "additionalProperties": False,
            "required": ["present", "mm", "direction"],
            "properties": {"present": PRESENCE, "mm": {"type": ["number", "null"]},
                           "direction": {"type": "string", "enum": ["left", "right", "na"]}},
        },
        "hydrocephalus": {
            "type": "object", "additionalProperties": False,
            "required": ["present"], "properties": {"present": PRESENCE},
        },
        "other": {
            "type": "object", "additionalProperties": False,
            "required": ["atrophy", "small_vessel_disease", "extra_axial_collection",
                         "pneumocephalus", "edema", "herniation", "devices", "postop",
                         "comparison"],
            "properties": {
                "atrophy": PRESENCE, "small_vessel_disease": PRESENCE,
                "extra_axial_collection": PRESENCE, "pneumocephalus": PRESENCE,
                "edema": PRESENCE, "herniation": PRESENCE,
                "devices": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
                "postop": PRESENCE,
                "comparison": {"type": "string",
                               "enum": ["none", "stable", "new", "increased", "decreased", "mixed"]},
            },
        },
    },
}

# gpt-oss-120b(reasoning=low)의 실측 오류를 프롬프트로 막는다. 손으로 확인한 90케이스 기준:
# 골절 지어냄 16→0/30, infarct acuity chronic 오판 29→1/30, chronic 후유증 present 누락 0.
# reasoning=high는 실제 골절을 오히려 억제해서 low를 쓴다.
SYSTEM_PROMPT = (
    "You are a precise neuroradiology information-extraction engine. Extract structured findings "
    "from a non-contrast head CT report into the given JSON schema. Output ONLY valid JSON.\n"
    "GENERAL\n"
    "- Set a finding's present to 'yes' ONLY if the report EXPLICITLY describes that finding. Never "
    "infer a finding from indirect context (trauma history, clinical indication, post-operative state, "
    "sinus/mastoid opacification, or another finding). If not mentioned, present='no'.\n"
    "- NEGATION is strict: 'no / without / no evidence of / negative for / free of / ruled out' => present='no'.\n"
    "- A finding described as known / residual / resolving / regressing / decreasing / unchanged / chronic "
    "is STILL on the scan => present='yes'. Only explicit full resolution or absence => 'no'.\n"
    "- HEDGING ('questionable','possible','suspected','cannot exclude','beginning','probable') => present='uncertain'.\n"
    "- Use 'na' for any attribute the report does not state; never guess laterality, location, or subtype.\n"
    "FRACTURE (do NOT over-call)\n"
    "- present='yes' ONLY if the report literally states a fracture ('fracture','fractured'). Do NOT infer a "
    "fracture from trauma, hematoma, surgery, sinus fluid, or facial findings. If the word fracture is absent, present='no'.\n"
    "INFARCT — separate PRESENCE from ACUITY\n"
    "- PRESENCE: set present='yes' for an infarct at ANY stage — an acute/subacute infarct, OR its chronic "
    "sequelae (encephalomalacia, gliosis, an old or lacunar defect, a cavitation or hypodensity that followed an "
    "infarct, post-ischemic changes). Never mark a described infarct or its old sequelae as 'na' or 'no'.\n"
    "- ACUITY: choose strictly from the stage the report describes, independently of presence:\n"
    "  * acute/subacute: 'demarcation','demarcated','recent','acute','new','increasingly hypodense','cytotoxic',"
    "'evolving','swelling/edema of the infarct'. NOTE: 'infarct demarcation' = an infarct just becoming visible "
    "= acute/subacute, NEVER chronic.\n"
    "  * chronic: 'old','chronic','established','encephalomalacia','gliosis','lacunar','microangiopathic',"
    "'old/cavitated defect','a defect/cavitation/hypodensity following an infarct','post-ischemic','residual scar'.\n"
    "ARRAYS\n"
    "- Never repeat a value in an array; list each location/subtype at most once.\n"
    "- hemorrhage.subtypes: include ONLY subtypes explicitly named; do not enumerate all types; 'contusion' only if stated.\n"
    "Output only valid JSON matching the schema."
)


SCHEMA_HINT = (
    "Output JSON with exactly these top-level keys: hemorrhage{present,subtypes[],laterality,"
    "locations[],acuity}, infarct{present,laterality,locations[],acuity}, fracture{present,location}, "
    "mass{present,laterality,locations[]}, mass_effect{present}, midline_shift{present,mm,direction}, "
    "hydrocephalus{present}, other{atrophy,small_vessel_disease,extra_axial_collection,pneumocephalus,"
    "edema,herniation,devices[],postop,comparison}. present=yes|no|uncertain."
)


def build_messages(report_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{SCHEMA_HINT}\n\nReport:\n{report_text}\n\nExtract findings as JSON."},
    ]
