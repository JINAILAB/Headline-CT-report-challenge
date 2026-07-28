#!/usr/bin/env bash
# 준비 3 — 재정렬 학습에 필요한 두 캐시.
#
#   text_cache : 리포트의 Qwen3-Embedding-8B 임베딩 (≈3 GB)
#   pe_cache   : frozen patch_embed 출력 (≈387 GB) — raw CT 재처리 병목을 없앤다
#                ⚠️ znorm=False로 만든다. 이 캐시로 학습된 encoder는 이후 캐싱도 znorm=False다.
#
# pe_cache는 scripts/01_train_encoder.sh 가 끝나면 지워도 된다.
#
# 사용:  GPU=0 bash scripts/prep/02_caches.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

LOG_DIR="$RUNS/prep"; mkdir -p "$LOG_DIR"

banner "text_cache (Qwen3-Embedding-8B)" "out = $HEADLINE_DATA_ROOT/interim/text_cache"
CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.prep.cache_text \
  2>&1 | tee -a "$LOG_DIR/cache_text.log"

banner "pe_cache (frozen patch_embed)" "out = $HEADLINE_DATA_ROOT/interim/pe_cache (≈387 GB)"
CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.prep.cache_patchembed \
  2>&1 | tee -a "$LOG_DIR/cache_patchembed.log"

banner "내부 val split 생성" "out = splits/vlm_val_patients.json"
$PY -u -m headline.splits

log "완료 — 이제 bash scripts/01_train_encoder.sh"
