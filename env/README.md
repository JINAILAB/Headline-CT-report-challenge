# Environments

Four isolated Python environments are required. They cannot be merged: Viola-Unet pins
`monai==1.5.0` (its network class signatures break on other versions, and it loads weights with
`strict=True`), vLLM pins its own `torch`/`transformers`, and TotalSegmentator pulls in a
separate nnU-Net stack.

| Environment | Used by | Install |
|---|---|---|
| **train** | everything except the three below — `headline.train.*`, `headline.data.*`, `headline.eval.generate`, `headline.eval.presence_f1`, `headline.prep.cache_*`, `headline.prep.project_labels`, `headline.prep.build_augmentation` | `uv sync` (from `pyproject.toml`), then `uv pip install flash-attn --no-build-isolation` and the `rve` package below |
| **seg** | `headline.prep.totalseg` | `uv tool install totalsegmentator` |
| **viola** | `headline.prep.viola` | `uv venv .venvs/viola && uv pip install -r env/requirements-viola.txt` |
| **vllm** | `headline.prep.extract_labels` (both reference and generated reports) | `uv venv .venvs/vllm && uv pip install -r env/requirements-vllm.txt` |

`requirements-{train,viola,vllm}.txt` are exact freezes of the environments used to produce the
reported numbers. `pyproject.toml` carries looser bounds for the train environment; use the
freeze if you want bit-level fidelity.

## Extra install steps for the train environment

**`rve` (rad-vision-engine)** — supplies the 11 CT display windows that Pillar-0 expects. It is
not on PyPI:

```bash
git clone -b release https://github.com/yalalab/rad-vision-engine
uv pip install -e rad-vision-engine
```

Without it, `headline.data.ct_input` cannot import and no CT can be encoded.

**flash-attn** is optional. `headline.models.vlm` falls back to `sdpa` automatically if the
import fails; training is slower but numerically equivalent.

## Pointing the scripts at each environment

Every script takes the interpreter from an environment variable, so nothing is hard-coded:

```bash
PY=.venvs/train/bin/python                                   # default for all scripts
TS_PY=$HOME/.local/share/uv/tools/totalsegmentator/bin/python  # scripts/prep/00_masks.sh
VIOLA_PY=.venvs/viola/bin/python                             # scripts/prep/00_masks.sh
VLLM_PY=.venvs/vllm/bin/python                               # scripts/prep/01_report_labels.sh, scripts/04_eval.sh
```

All of them need `PYTHONPATH=src`, which `scripts/common.sh` sets for you.
