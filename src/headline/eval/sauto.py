from __future__ import annotations

from typing import Dict, List

WEIGHTS = {"bertscore_f1": 0.7, "meteor": 0.1, "rouge_l": 0.1, "bleu4": 0.1}


def _bertscore_f1(preds: List[str], refs: List[str]) -> float:
    from bert_score import score
    _, _, f1 = score(preds, refs, lang="en", verbose=False, rescale_with_baseline=True)
    return f1.mean().item()


def _meteor(preds: List[str], refs: List[str]) -> float:
    from nltk.translate.meteor_score import meteor_score
    return sum(meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)) / len(preds)


def _rouge_l(preds: List[str], refs: List[str]) -> float:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sum(sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)


def _bleu4(preds: List[str], refs: List[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    sm = SmoothingFunction().method1
    return sum(sentence_bleu([r.split()], p.split(), weights=(.25, .25, .25, .25),
                             smoothing_function=sm) for p, r in zip(preds, refs)) / len(preds)


def compute_sauto(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """생성문/정답 리스트 → 지표 dict + sauto. 개별 지표 실패는 NaN으로 격리"""
    assert len(preds) == len(refs) and preds, "preds/refs 길이 불일치 또는 빈 입력"
    preds = [(p or "").strip() or "." for p in preds]     # 빈 생성 시 라이브러리 크래시 방지
    refs = [r.strip() for r in refs]
    out: Dict[str, float] = {}
    for name, fn in [("bertscore_f1", _bertscore_f1), ("meteor", _meteor),
                     ("rouge_l", _rouge_l), ("bleu4", _bleu4)]:
        try:
            out[name] = float(fn(preds, refs))
        except Exception as e:                            # noqa: BLE001 — 한 지표 실패로 전체를 막지 않는다
            out[name] = float("nan")
            out[f"{name}_err"] = type(e).__name__
    out["sauto"] = sum(w * (out[k] if out[k] == out[k] else 0.0) for k, w in WEIGHTS.items())
    return out
