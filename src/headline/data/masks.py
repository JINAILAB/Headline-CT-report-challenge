from __future__ import annotations

from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .ct_input import TARGET_DHW

BRAIN_REGION_LABELS = list(range(1, 17))          # TS brain_structures 1..16
N_REGIONS = len(BRAIN_REGION_LABELS) + 2          # +bone +hemorrhage = 18
REGION_NAMES: List[str] = [
    "brainstem", "subarachnoid_space", "venous_sinuses", "septum_pellucidum", "cerebellum",
    "caudate_nucleus", "lentiform_nucleus", "insular_cortex", "internal_capsule", "ventricle",
    "central_sulcus", "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe",
    "thalamus", "craniofacial_bone", "hemorrhage",
]
assert len(REGION_NAMES) == N_REGIONS

MASK_FILES = {
    "brain_regions": "brain_regions.nii.gz",
    "bone_craniofacial": "bone_craniofacial.nii.gz",
    "hemorrhage_viola": "hemorrhage_viola.nii.gz",
}


def _nii_to_dhw(path: Path) -> torch.Tensor:
    arr = np.asarray(nib.load(str(path)).dataobj)                 # (H,W,D)
    return torch.as_tensor(arr).permute(2, 0, 1).contiguous().float()   # (D,H,W)


def _raw_channels(mask_dir: Path) -> List[torch.Tensor]:
    """18개 채널의 네이티브 해상도 이진 볼륨. 소스가 없으면 스칼라 placeholder를 넣는다."""
    mask_dir = Path(mask_dir)
    out: List[torch.Tensor] = []

    br_path = mask_dir / MASK_FILES["brain_regions"]
    br = _nii_to_dhw(br_path) if br_path.exists() else None
    for lab in BRAIN_REGION_LABELS:
        out.append((br == lab).float() if br is not None else torch.zeros(1))

    for key in ("bone_craniofacial", "hemorrhage_viola"):
        p = mask_dir / MASK_FILES[key]
        out.append((_nii_to_dhw(p) > 0).float() if p.exists() else torch.zeros(1))
    return out


def _resize_nearest(m: torch.Tensor, size) -> torch.Tensor:
    if m.dim() != 3:                                       # 소스 없음 → 0 볼륨
        return torch.zeros(size)
    return F.interpolate(m[None, None], size=size, mode="nearest")[0, 0]


def load_region_masks(mask_dir: Path) -> torch.Tensor:
    """→ (18, 128, 256, 256) 이진 마스크. 없는 소스 채널은 0."""
    masks = torch.stack([_resize_nearest(m, TARGET_DHW) for m in _raw_channels(mask_dir)], 0)
    assert masks.shape == (N_REGIONS, *TARGET_DHW), masks.shape
    return masks


def load_region_masks32(mask_dir: Path, grid=(32, 32, 32)) -> torch.Tensor:
    """→ (18, 32, 32, 32) uint8 — feature 캐시에 저장하는 형태.
    """
    chans = [_resize_nearest(_resize_nearest(m, TARGET_DHW), grid) for m in _raw_channels(mask_dir)]
    masks = torch.stack(chans, 0).to(torch.uint8)
    assert masks.shape == (N_REGIONS, *grid), masks.shape
    return masks
