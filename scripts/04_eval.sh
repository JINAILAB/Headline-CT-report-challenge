#!/usr/bin/env bash
# 4단계 — held-out 600 스터디 평가 (논문 Table 2).
#
#   (a) 리포트 생성 + S_Auto            [학습 환경]
#   (b) 생성 리포트에서 gpt-oss 소견 추출 [vLLM 환경]
#   (c) 참조 라벨과 대칭 CORE7 presence-F1
#
# 두 환경을 나누는 이유: vLLM과 학습 스택은 torch 핀이 충돌하고, 한 프로세스에서 7B 생성과
# 120B 추출을 같이 올릴 수도 없다.
#
# 사용:  GPU=0 VLLM_PY=.venvs/vllm/bin/python bash scripts/04_eval.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CKPT="${CKPT:-$STAGE2_OUT/ckpt_stage2_best.pt}"
VLLM_PY="${VLLM_PY:-$PY}"
GOLD_ROOT="${GOLD_ROOT:-$HEADLINE_DATA_ROOT/interim/report_labels/gptoss}"
GEN_CSV="$EVAL_OUT/generated.csv"
PRED_DIR="$EVAL_OUT/generated_findings"

test -f "$CKPT" || { log "$CKPT 없음 — 먼저 scripts/03_train_stage2.sh"; exit 1; }
mkdir -p "$EVAL_OUT"

banner "EVAL 1/3 — 600 스터디 생성 + S_Auto" \
  "ckpt       = $CKPT" \
  "cache_root = $CACHE_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.eval.generate \
  --ckpt "$CKPT" --cache_root "$CACHE_ROOT" --out "$GEN_CSV" \
  2>&1 | tee -a "$EVAL_OUT/generate.log"

banner "EVAL 2/3 — 생성 리포트에서 소견 추출 (gpt-oss-120b)" "out = $PRED_DIR"
CUDA_VISIBLE_DEVICES="$GPU" $VLLM_PY -u -m headline.prep.extract_labels \
  --source csv --gen_csv "$GEN_CSV" --out_dir "$PRED_DIR" --schema core \
  2>&1 | tee -a "$EVAL_OUT/extract.log"

banner "EVAL 3/3 — CORE7 presence-F1 (대칭)" "gold = $GOLD_ROOT"
$PY -u -m headline.eval.presence_f1 \
  --pred_dir "$PRED_DIR" --gold_root "$GOLD_ROOT" --out "$EVAL_OUT/presence_f1.json" \
  2>&1 | tee -a "$EVAL_OUT/presence_f1.log"

log "완료 — $EVAL_OUT/{generated.sauto.json, presence_f1.json}"
