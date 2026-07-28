#!/usr/bin/env bash
# 2단계 — report generator stage 1: LLM은 얼리고 visual tokenizer만 학습.
#
# 사용:  GPU=0 bash scripts/02_train_stage1.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "STAGE 1 — visual tokenizer 정렬" \
  "config     = configs/stage1.json" \
  "cache_root = $CACHE_ROOT" \
  "out        = $STAGE1_OUT"

test -d "$CACHE_ROOT" || { log "$CACHE_ROOT 없음 — 먼저 scripts/01_train_encoder.sh"; exit 1; }
mkdir -p "$STAGE1_OUT"

CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.train.reportgen \
  --config configs/stage1.json \
  --cache_root "$CACHE_ROOT" \
  --out "$STAGE1_OUT" \
  "$@" 2>&1 | tee -a "$STAGE1_OUT/train.log"

test -f "$STAGE1_OUT/ckpt_stage1.pt" || { log "stage 1 실패"; exit 1; }
log "완료 → $STAGE1_OUT/ckpt_stage1.pt"
log "다음: bash scripts/03_train_stage2.sh"
