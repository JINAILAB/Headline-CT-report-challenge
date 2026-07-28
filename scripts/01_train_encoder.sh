#!/usr/bin/env bash
# 1단계 — encoder contrastive re-alignment (논문 Fig.1(a)) + 그 결과로 feature 재캐싱.
#
#   (a) aligner 학습      : pe_cache + text_cache + presence 라벨 → CORE7 AUROC 최고 체크포인트
#   (b) encoder export    : enc.* 만 추출 (znorm 계약 포함)
#   (c) feature 재캐싱    : pe_cache → 재정렬 encoder → pillar_cache_realigned  (≈402 GB)
#
# 사전조건: scripts/prep/* 을 모두 돌려 masks / report_labels / classifier_labels /
#           text_cache / pe_cache 가 준비돼 있어야 한다.
#
# 사용:  GPU=0 CACHE_GPUS=0,1 bash scripts/01_train_encoder.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CACHE_GPUS="${CACHE_GPUS:-$GPU}"        # 재캐싱에 쓸 GPU 목록(쉼표 구분)
ALIGNER_DIR="$ENCODER_OUT/aligner"
ENC_CKPT="$ENCODER_OUT/enc_realigned.pt"

banner "STEP 1/3 — aligner 학습" \
  "config     = configs/realign.json" \
  "out        = $ALIGNER_DIR"
CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.train.realign \
  --config configs/realign.json --out "$ALIGNER_DIR" 2>&1 | tee -a "$ENCODER_OUT/aligner.log"

test -f "$ALIGNER_DIR/aligner_bestauroc.pt" || { log "aligner 실패"; exit 1; }

banner "STEP 2/3 — encoder export" "out = $ENC_CKPT"
CUDA_VISIBLE_DEVICES="$GPU" $PY -u -m headline.train.export_encoder \
  --ckpt "$ALIGNER_DIR/aligner_bestauroc.pt" --out "$ENC_CKPT"

banner "STEP 3/3 — feature 재캐싱" \
  "source = pe_cache (재정렬 encoder, znorm은 체크포인트에서 자동)" \
  "out    = $CACHE_ROOT" \
  "gpus   = $CACHE_GPUS"
IFS=',' read -ra GPU_LIST <<< "$CACHE_GPUS"
NSHARDS="${#GPU_LIST[@]}"
for split in batch1 batch2 batch3; do
  for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" $PY -u -m headline.data.cache_features \
      --split "$split" --source pe_cache --enc_ckpt "$ENC_CKPT" \
      --out_root "$CACHE_ROOT" --shard "$i" --nshards "$NSHARDS" \
      >> "$ENCODER_OUT/cache_${split}_sh${i}.log" 2>&1 &
  done
  wait
  log "재캐싱 완료: $split"
done

N=$(find "$CACHE_ROOT" -name '*.pt' | wc -l)
log "완료. encoder=$ENC_CKPT, 캐시 $N개 → $CACHE_ROOT"
log "다음: bash scripts/02_train_stage1.sh"
