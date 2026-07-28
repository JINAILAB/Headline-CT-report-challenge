#!/usr/bin/env bash
# 준비 1 — 해부/출혈 마스크 (region tokenizer의 18채널 소스).
#
#   TotalSegmentator brain_structures        → brain_regions.nii.gz      (채널 0..15)
#   TotalSegmentator craniofacial_structures → bone_craniofacial.nii.gz  (채널 16)
#   Viola-Unet                                → hemorrhage_viola.nii.gz   (채널 17)
#
# 두 도구는 서로 다른 전용 환경이 필요하다(docs/DATA.md 참고).
#   TS_PY    — TotalSegmentator가 설치된 python (brain_structures는 학술 라이선스 키 필요)
#   VIOLA_PY — monai==1.5.0으로 고정된 python
# 모두 재개 가능하다(이미 만들어진 출력은 건너뛴다).
#
# 사용:  TS_PY=... VIOLA_PY=... GPUS=0,1 bash scripts/prep/00_masks.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

TS_PY="${TS_PY:-$PY}"
VIOLA_PY="${VIOLA_PY:-$PY}"
GPUS="${GPUS:-$GPU}"
WORKERS="${WORKERS:-8}"
VIOLA_SHARDS="${VIOLA_SHARDS:-4}"
SPLITS="${SPLITS:-batch1 batch2 batch3}"
LOG_DIR="$RUNS/prep"; mkdir -p "$LOG_DIR"

for split in $SPLITS; do
  banner "TotalSegmentator — $split" "gpus=$GPUS workers=$WORKERS"
  $TS_PY -u -m headline.prep.totalseg --split "$split" --gpus "$GPUS" --workers "$WORKERS" \
    2>&1 | tee -a "$LOG_DIR/totalseg_${split}.log"
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
for split in $SPLITS; do
  banner "Viola-Unet — $split" "shards=$VIOLA_SHARDS"
  for ((i = 0; i < VIOLA_SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" \
      $VIOLA_PY -u -m headline.prep.viola --split "$split" \
        --shard "$i" --nshards "$VIOLA_SHARDS" >> "$LOG_DIR/viola_${split}_sh${i}.log" 2>&1 &
    sleep 8    # 앙상블 로드가 겹치지 않게 stagger
  done
  wait
done

log "완료 → $HEADLINE_DATA_ROOT/interim/masks"
