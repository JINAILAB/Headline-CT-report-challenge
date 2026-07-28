# Data preparation

This is the part that does not reproduce itself. Training is three shell scripts; getting to the
point where you can run them means producing three families of artefacts:

| Artefact | Produced by | Used by | Size |
|---|---|---|---|
| Anatomical + haemorrhage masks | TotalSegmentator, Viola-Unet | region tokeniser (18 channels) | ~12 GB |
| Structured findings + presence labels | gpt-oss-120b via vLLM | re-alignment BCE, stage-2 task mix, **and evaluation** | ~300 MB |
| Text and patch-embed caches | Qwen3-Embedding-8B, frozen Pillar-0 conv stem | re-alignment | ~390 GB |

Everything below assumes `HEADLINE_DATA_ROOT` is set and `PYTHONPATH=src`
(`scripts/common.sh` does both).

---

## 1. Masks — `scripts/prep/00_masks.sh`

### 1.1 TotalSegmentator

Two tasks are run per study, both on the **soft-tissue (`WT`) NIfTI**:

| Task | Output | Licence |
|---|---|---|
| `brain_structures` | `brain_regions.nii.gz`, labels 1–16 | **academic, non-commercial — needs a licence key** |
| `craniofacial_structures` | `bone_craniofacial.nii.gz`, labels 1–7 | Apache-2.0 |

```bash
uv tool install totalsegmentator
totalseg_set_license -l aca_XXXXXXXXXXXX      # required for brain_structures
```

Request the academic licence at <https://backend.totalsegmentator.com/license-academic/>. Without
it `brain_structures` refuses to run and channels 0–15 of the region tokeniser are all zero. This
is the single most likely thing to block a reproduction.

nnU-Net weights download automatically on first use into `~/.totalsegmentator/nnunet/results`.

**Why no resampling step.** `TotalSegmentator --ml` resamples its own 1.5 mm output back onto the
*input* grid before writing. Feeding it the WT volume therefore yields masks already aligned to
the WT grid, which is what makes them intersectable with the Viola output and poolable over the
Pillar feature grid.

**Concurrency.** The orchestrator in `headline/prep/totalseg.py` runs one subprocess per study
with GPU round-robin, a staggered launch (CUDA init storms otherwise), retries, and skip-if-exists
resume. It passes `-nr 1 -ns 1` (single resample/save thread): with the defaults, dozens of
concurrent workers exceed the container's `pids.max` and fail with `BlockingIOError`. Keep
`workers × threads` near the core count.

Two TotalSegmentator tasks that appear in some earlier notes are **not** used: `total` (only
`brain` and `skull` are non-empty in a head FOV, and neither feeds the 18 channels) and
`headneck_bones_vessels` (always empty in a head CT FOV).

### 1.2 Viola-Unet

Produces `hemorrhage_viola.nii.gz` (`0 bg, 1 EDH, 2 SDH, 3 SAH, 4 IPH, 5 IVH`) and a per-study
CSV of subtype probabilities and volumes. Channel 17 of the region tokeniser is this mask
binarised; the subtype identity is not used by the tokeniser.

**Getting the code and weights.** Upstream is <https://github.com/samleoqh/viola3_docker>, published
as the `violai:3.1` Docker image. The weights (`ckpt/models/model{1..5}/*.bread`, ~280 MB) are
**CC BY-NC-ND**, so they are not vendored here and may not be redistributed. Extract them from the
image into `third_party/violai/` so that `main.py` and `ckpt/` sit side by side:

```bash
docker create --name viola violai:3.1
docker cp viola:/app/. third_party/violai/
docker rm viola
uv venv .venvs/viola && uv pip install -r env/requirements-viola.txt
```

`monai==1.5.0` is a hard pin: the network class signatures change between MONAI versions and the
checkpoints are loaded with `strict=True`, so any other version raises on load.

**What it is.** A five-model ensemble — one DynUNet/ViolaUNet and two RL variants (3-channel
windowed input, binary), one PlainViolaConvUNet (1-channel, binary), and one ResidualEncoderUNet
(1-channel, 6-class) that supplies the subtype distribution. The four binary models are averaged,
fused with the subtype head, and thresholded at 0.5 (relaxed to 0.45/0.4 on conflict). Each model
family has its own MONAI preprocessing chain (intensity windowing → `Spacingd` to 1×1×3 mm →
`CropForegroundd` → `Orientationd` RPI or RAS), and each is undone by `Invertd` afterwards; the
writer then copies the original affine and **raises on any shape mismatch**, which is what
guarantees the output lands on the WT grid.

**Known limitations, straight from upstream.** Input must be non-contrast axial CT. At 4–5 mm
slice thickness with the skull present there is a false-positive tendency at the skull boundary
(EDH/SDH over-segmented; IPH/IVH accurate). The CSV's `Prob_any_ICH` is a voxel max-pool and
saturates around 0.909 — treat only the masks and volumes as usable.

**Batching.** `main.py` loads the ensemble once (~30 s) then streams a folder at ~3 s/study, so
`headline/prep/viola.py` shards the studies and runs it **once per shard**, symlinking each WT
volume into a flat temp directory as `<study_id>.nii.gz` and distributing the outputs afterwards.

### 1.3 The 18 channels

Assembled in `headline/data/masks.py`, which is the authoritative definition:

| Channel | Source |
|---|---|
| 0–15 | `brain_regions.nii.gz` labels 1–16: brainstem, subarachnoid space, venous sinuses, septum pellucidum, cerebellum, caudate, lentiform nucleus, insular cortex, internal capsule, ventricle, central sulcus, frontal, parietal, occipital, temporal lobe, thalamus |
| 16 | `bone_craniofacial.nii.gz > 0` |
| 17 | `hemorrhage_viola.nii.gz > 0` |

No left/right split — `brain_structures` does not provide one. Channel 16 is described as a skull
mask in the paper; strictly it is the whole craniofacial multilabel binarised, so it also contains
mandible, teeth and sinuses. A missing source file yields a zero channel rather than an error, so
partial preparation degrades gracefully instead of crashing.

---

## 2. Structured findings — `scripts/prep/01_report_labels.sh`

Runs `gpt-oss-120b` locally under vLLM with JSON-schema-constrained decoding. Nothing leaves the
machine; no external API is involved, which the data use agreement requires.

Two passes over every report:

* **core** (`headline/prep/report_schema.py`) — the seven core findings plus attributes
  (haemorrhage subtype and acuity, infarct acuity, laterality, location, midline shift in mm, …),
  each with `present ∈ {yes, no, uncertain}`. → `report_labels/gptoss/`
* **detailed** (`headline/prep/report_schema_detailed.py`) — 24 finding categories with the verbatim
  source clause, plus a 9-axis `normal_survey`. → `report_labels/gptoss_detailed/`

`headline/prep/project_labels.py` then merges them into the fixed 20-key presence vector
(`headline/labels.py`). The merge is deliberately asymmetric:

* CORE7 presence comes **only** from the core schema. The detailed schema's per-finding `present`
  marks normal statements ("midline central", "sinuses clear") as positive and cannot be used here.
* `sinus_mastoid` and `orbit` come from `normal_survey`, which *does* distinguish normal from
  abnormal.
* `calcification`, `scalp_soft_tissue`, `vascular` come from the detailed categories, since those
  are only ever mentioned when present.

Sampling is deterministic (`temperature=0`) with `reasoning_effort=low`. Higher reasoning effort
was measurably worse — it suppresses genuine fractures. The prompt encodes several failure modes
found by hand-checking 90 cases; see the comments in `report_schema.py` before editing it.

Output is one JSON per study; the extractor skips studies that already have one, so it restarts
cleanly. Parse failures are stored as `{"_parse_error": …}` and dropped downstream rather than
crashing the run.

`headline/prep/build_augmentation.py` then turns the same labels into the stage-2 instruction mix
(one report-writing task per study plus Bernoulli-sampled auxiliary tasks).

> The evaluation in `scripts/04_eval.sh` reuses `report_labels/gptoss/` as its gold. If you change
> the extraction model or prompt, regenerate both sides or the presence-F1 stops being symmetric.

---

## 3. Caches — `scripts/prep/02_caches.sh`

* `text_cache/` (~3 GB) — Qwen3-Embedding-8B embeddings of each report: the whole Findings section
  (global alignment) and each sentence (local alignment). fp32 model, attention-masked mean pool,
  L2 normalised, `max_length=512`, no instruction prefix — the recipe Pillar-0 itself was trained
  with.
* `pe_cache/` (~387 GB) — the frozen conv stem's output, `(384, 32, 32, 32)` fp16 per study.
  Raw CT is 369 MB per study and NIfTI decode plus windowing is ~94 % of data-loading time; the
  conv stem is frozen anyway, so caching its output makes re-alignment training practical. **This
  is deletable once `scripts/01_train_encoder.sh` has finished.**

This step also writes `splits/vlm_val_patients.json` — the internal validation split, derived
deterministically (patient-level, `seed=42`) from the studies that have reports. The file itself is
not redistributed here; `splits/vlm_val_patients.example.json` shows only the expected format (a
JSON list of patient IDs). Generate it with the prep step above, or directly:

```bash
uv run python -m headline.splits
```

---