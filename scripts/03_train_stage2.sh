#!/usr/bin/env bash
# 3단계 — report generator stage 2: LoRA 추가 + augmentation 태스크 믹스.
# held-out 600 스터디의 생성 S_Auto로 best 체크포인트를 고른다.
#
# 사용:  GPU=0 bash scripts/03_train_stage2.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

INIT_FROM="${INIT_FROM:-$STAGE1_OUT/ckpt_stage1.pt}"

banner "STAGE 2 — LoRA + augmentation" \
  "config     = configs/stage2.json" \
  "init_from  = $INIT_FROM" \
  "cache_root = $CACHE_ROOT" \
  "out        = $STAGE2_OUT"

test -f "$INIT_FROM" || { log "$INIT_FROM 없음 — 먼저 scripts/02_train_stage1.sh"; exit 1; }
mkdir -p "$STAGE2_OUT"

CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.train.reportgen \
  --config configs/stage2.json \
  --cache_root "$CACHE_ROOT" \
  --init_from "$INIT_FROM" \
  --out "$STAGE2_OUT" \
  "$@" 2>&1 | tee -a "$STAGE2_OUT/train.log"

test -f "$STAGE2_OUT/ckpt_stage2_best.pt" || { log "stage 2 실패"; exit 1; }
log "완료 → $STAGE2_OUT/ckpt_stage2_best.pt (best-by-S_Auto)"
log "다음: bash scripts/04_eval.sh"
