#!/usr/bin/env bash
# 준비 2 — 리포트에서 구조화 소견 추출 (로컬 vLLM + gpt-oss-120b).
#
#   core     스키마 → report_labels/gptoss/<split>/       (CORE7 + 속성; 평가의 gold이기도 하다)
#   detailed 스키마 → report_labels/gptoss_detailed/      (normal_survey 9축 + 세부 category)
# 그다음 두 결과를 합쳐 20차원 presence 라벨을 만든다.
#
# ⚠️ 평가(scripts/04_eval.sh)가 gptoss/ 를 gold로 쓰므로 **같은 모델·프롬프트**를 유지해야
#    presence-F1이 대칭이 된다.
#
# 사용:  VLLM_PY=.venvs/vllm/bin/python GPU=0 bash scripts/prep/01_report_labels.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VLLM_PY="${VLLM_PY:-$PY}"
MODEL="${MODEL:-$REPO_ROOT/models/gpt-oss-120b}"
LABELS="$HEADLINE_DATA_ROOT/interim/report_labels"
SPLITS="${SPLITS:-batch1 batch2 batch3}"
LOG_DIR="$RUNS/prep"; mkdir -p "$LOG_DIR"

for split in $SPLITS; do
  banner "소견 추출 (core) — $split" "model=$MODEL"
  CUDA_VISIBLE_DEVICES="$GPU" $VLLM_PY -u -m headline.prep.extract_labels \
    --model "$MODEL" --source reports --split "$split" --schema core \
    --out_dir "$LABELS/gptoss/$split" 2>&1 | tee -a "$LOG_DIR/labels_core_${split}.log"

  banner "소견 추출 (detailed) — $split"
  CUDA_VISIBLE_DEVICES="$GPU" $VLLM_PY -u -m headline.prep.extract_labels \
    --model "$MODEL" --source reports --split "$split" --schema detailed \
    --out_dir "$LABELS/gptoss_detailed/$split" 2>&1 | tee -a "$LOG_DIR/labels_detailed_${split}.log"
done

banner "presence 라벨 투영" "out = $HEADLINE_DATA_ROOT/processed/classifier_labels"
$PY -u -m headline.prep.project_labels 2>&1 | tee -a "$LOG_DIR/project_labels.log"

banner "stage 2 augmentation 인스턴스 생성" "out = $HEADLINE_DATA_ROOT/processed/augment"
$PY -u -m headline.prep.build_augmentation 2>&1 | tee -a "$LOG_DIR/build_augmentation.log"

log "완료"
