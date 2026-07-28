#!/usr/bin/env bash
# 모든 스크립트가 source 하는 공통 설정. 값은 전부 환경변수로 덮어쓸 수 있다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HEADLINE_DATA_ROOT="${HEADLINE_DATA_ROOT:-$REPO_ROOT/data}"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$REPO_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"       # 모든 가중치는 로컬(오프라인 재현)
export WANDB_MODE="${WANDB_MODE:-offline}"

# 학습/평가 환경의 python. uv를 쓰면 `PY="uv run python"`.
PY="${PY:-python}"
GPU="${GPU:-0}"

RUNS="${RUNS:-$REPO_ROOT/runs}"
ENCODER_OUT="${ENCODER_OUT:-$RUNS/encoder}"
STAGE1_OUT="${STAGE1_OUT:-$RUNS/stage1}"
STAGE2_OUT="${STAGE2_OUT:-$RUNS/stage2}"
EVAL_OUT="${EVAL_OUT:-$RUNS/eval}"
# 재정렬 encoder로 만든 Pillar feature 캐시(1단계 산출물, 이후 학습·평가의 입력)
CACHE_ROOT="${CACHE_ROOT:-$HEADLINE_DATA_ROOT/interim/pillar_cache_realigned}"

log() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*"; }

banner() {
  log "$1"
  shift || true
  for kv in "$@"; do printf '        %s\n' "$kv"; done
}
