# HEADLINE: Finding-Aware Head CT Report Generation

Reference implementation of our system for the MICCAI 2026 HEADLINE Challenge.

This repository contains code for:

* realigning a pretrained head CT encoder with radiology reports;
* training visual tokenization modules for report generation;
* fine-tuning Qwen2.5-7B-Instruct with LoRA; and
* evaluating generated reports using S_Auto and CORE7 presence-F1.

## Overview

The pipeline consists of four stages:

1. Prepare region masks and report-derived finding labels.
2. Realign the Pillar-0 encoder with radiology reports.
3. Train the report-generation model.
4. Generate reports and compute evaluation metrics.

```text
HEADLINE data
    │
    ├── Anatomical and hemorrhage masks
    ├── Structured finding labels
    ├── Text embedding cache
    └── Patch embedding cache
            │
            ▼
    Encoder realignment
            │
            ▼
    Visual tokenization training
            │
            ▼
    LoRA fine-tuning
            │
            ▼
    Report generation and evaluation
```

The report generator uses:

* a frozen or realigned Pillar-0 head CT encoder;
* a Perceiver resampler;
* a region tokenizer; and
* Qwen2.5-7B-Instruct with LoRA.

## Requirements

The complete pipeline requires one or more NVIDIA H100 GPUs (recommended).

The preprocessing tools have incompatible dependency requirements and should therefore be installed in isolated environments.

See [`env/README.md`](env/README.md) for environment-specific instructions.

## Installation

Install the main training and evaluation environment:

```bash
uv sync
```

FlashAttention is optional:

```bash
uv pip install flash-attn --no-build-isolation
```

When FlashAttention is unavailable, the code falls back to PyTorch SDPA.

Install `rad-vision-engine`:

```bash
git clone --branch release \
  https://github.com/yalalab/rad-vision-engine

uv pip install -e rad-vision-engine
```

Additional isolated environments are required for:

* TotalSegmentator;
* Viola-Unet; and
* vLLM-based finding extraction.

See [`env/README.md`](env/README.md) for details.

## Data

Set `HEADLINE_DATA_ROOT` to the HEADLINE dataset directory:

```bash
export HEADLINE_DATA_ROOT=/path/to/headline
```

Using a symbolic link is also supported:

```bash
ln -s /path/to/headline ./data
```

The expected directory structure is:

```text
$HEADLINE_DATA_ROOT/
└── raw/
    ├── batch1/
    │   ├── CT_data/
    │   │   └── <PatientID>/
    │   │       └── <StudyID>/
    │   │           └── <reconstruction>/
    │   │               └── *.nii.gz
    │   └── reports_train_batch1.csv
    ├── batch2/
    └── batch3/
```

The pipeline uses only the soft-tissue reconstruction whose directory name begins with `WT`. Bone-kernel reconstructions beginning with `KF` are ignored.

All derived-data paths are managed through `src/headline/paths.py`.

See [`docs/DATA.md`](docs/DATA.md) for detailed preparation instructions and batch-specific report formats.

## External Checkpoints and Tools

The following models and external tools are required:

| Component                | Default location              | Purpose                                       |
| ------------------------ | ----------------------------- | --------------------------------------------- |
| Pillar-0 head CT encoder | `refs/pillar0_headct/`        | CT feature extraction and encoder realignment |
| Qwen2.5-7B-Instruct      | `models/qwen2.5-7b-instruct/` | Report generation                             |
| Qwen3-Embedding-8B       | `models/qwen3-embedding-8b/`  | Report embedding during encoder realignment   |
| gpt-oss-120b             | `models/gpt-oss-120b/`        | Finding extraction and evaluation             |
| TotalSegmentator         | Installed separately          | Anatomical region masks                       |
| Viola-Unet               | `third_party/violai/`         | Hemorrhage masks                              |

The Pillar-0 location can also be specified through an environment variable:

```bash
export HEADLINE_PILLAR0=/path/to/pillar0
```

Model and dataset licenses are not included in this repository. Some components require gated access or separate license approval.

See [`docs/DATA.md`](docs/DATA.md) for download instructions and license information.

## Data Preparation

### 1. Generate Region Masks

```bash
TS_PY=/path/to/totalsegmentator/python \
VIOLA_PY=/path/to/viola/python \
GPUS=0,1 \
bash scripts/prep/00_masks.sh
```

This step creates 18 region masks using TotalSegmentator and Viola-Unet.

### 2. Extract Structured Findings

```bash
VLLM_PY=/path/to/vllm/python \
GPU=0 \
bash scripts/prep/01_report_labels.sh
```

This step converts free-text reports into structured finding labels and prepares the stage-2 task mixture.

### 3. Build Caches

```bash
GPU=0 \
bash scripts/prep/02_caches.sh
```

This step creates:

* report embedding caches using Qwen3-Embedding-8B; and
* patch embedding caches using the frozen Pillar-0 patch embedding module.

## Training

### 1. Realign the CT Encoder

```bash
GPU=0 \
CACHE_GPUS=0,1 \
bash scripts/01_train_encoder.sh
```

### 2. Train the Visual Tokenization Modules

```bash
GPU=0 \
bash scripts/02_train_stage1.sh
```

### 3. Fine-Tune with LoRA

```bash
GPU=0 \
bash scripts/03_train_stage2.sh
```

```bash
bash scripts/03_train_stage2.sh --epochs 2
```

## Evaluation

Generate reports and compute the evaluation metrics:

```bash
GPU=0 \
VLLM_PY=/path/to/vllm/python \
bash scripts/04_eval.sh
```

Evaluation outputs are written to:

```text
runs/eval/generated.sauto.json
runs/eval/presence_f1.json
```

## Smoke Test

The main training entry points support `--limit N` for small development runs.

For example:

```bash
uv run python -m headline.train.reportgen \
  --config configs/stage1.json \
  --limit 64 \
  --out runs/smoke
```

Run the test suite with:

```bash
uv run pytest
```

The smoke test verifies the main data-loading and report-generation code paths without building the complete training caches.

## Documentation

* [`docs/DATA.md`](docs/DATA.md): dataset preparation, checkpoints, and licenses
* [`env/README.md`](env/README.md): isolated preprocessing environments

## License

The source code is distributed under the terms described in [`LICENSE`](LICENSE).

The HEADLINE dataset, pretrained model checkpoints, TotalSegmentator, and Viola-Unet are governed by their respective licenses. Their data and weights are not redistributed through this repository.
