from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from headline.data.ct_input import TARGET_DHW
from headline.data.masks import (BRAIN_REGION_LABELS, MASK_FILES, N_REGIONS, REGION_NAMES,
                                 load_region_masks, load_region_masks32)
from headline.data.samplers import LengthGroupedBatchSampler


def _write_masks(tmp_path, shape=(16, 20, 12)):
    """(H,W,D) 네이티브 마스크 3종을 만든다. brain_regions는 label 1..16을 슬라이스로 채운다."""
    brain = np.zeros(shape, dtype=np.int16)
    for i, lab in enumerate(BRAIN_REGION_LABELS):
        brain[i, :, :] = lab                       # 각 라벨이 정확히 한 행
    bone = np.zeros(shape, dtype=np.int16)
    bone[:, 0, :] = 3                              # craniofacial 멀티라벨 값 하나
    hem = np.zeros(shape, dtype=np.int16)
    hem[:, :, 0] = 4                               # Viola IPH
    affine = np.eye(4)
    for name, arr in [("brain_regions", brain), ("bone_craniofacial", bone),
                      ("hemorrhage_viola", hem)]:
        nib.save(nib.Nifti1Image(arr, affine), tmp_path / MASK_FILES[name])
    return tmp_path


def test_region_names_match_channel_count():
    assert N_REGIONS == 18 == len(REGION_NAMES)


def test_load_region_masks_shape_and_binary(tmp_path):
    m = load_region_masks(_write_masks(tmp_path))
    assert m.shape == (N_REGIONS, *TARGET_DHW)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


def test_each_brain_label_becomes_its_own_channel(tmp_path):
    m = load_region_masks(_write_masks(tmp_path))
    for i in range(len(BRAIN_REGION_LABELS)):
        assert m[i].sum() > 0, f"channel {i} ({REGION_NAMES[i]}) is empty"


def test_bone_and_hemorrhage_channels_are_binarised(tmp_path):
    m = load_region_masks(_write_masks(tmp_path))
    assert m[16].sum() > 0 and m[17].sum() > 0     # 라벨 값 3, 4 → 1로


def test_missing_source_yields_zero_channels(tmp_path):
    m = load_region_masks(tmp_path)                # 파일 없음
    assert m.shape == (N_REGIONS, *TARGET_DHW)
    assert m.sum() == 0


def test_masks32_matches_two_step_downsample(tmp_path):
    """캐시에 저장하는 (18,32,32,32)가 full-res 경유 결과와 정확히 같아야 한다."""
    d = _write_masks(tmp_path)
    full = load_region_masks(d)
    ref = torch.nn.functional.interpolate(full[None], size=(32, 32, 32), mode="nearest")[0]
    got = load_region_masks32(d)
    assert got.shape == (N_REGIONS, 32, 32, 32)
    assert torch.equal(got, ref.to(torch.uint8))


def test_sampler_respects_token_budget():
    lengths = [50, 400, 1200, 90, 2000, 300, 700, 120] * 12
    g = torch.Generator().manual_seed(0)
    sampler = LengthGroupedBatchSampler(lengths, max_tokens=3072, max_bs=8,
                                        megabatch_size=32, generator=g)
    seen = []
    for batch in sampler:
        seen.extend(batch)
        assert len(batch) <= 8
        longest = max(lengths[i] for i in batch)
        assert len(batch) == 1 or len(batch) * longest <= 3072
    assert sorted(seen) == list(range(len(lengths)))     # 모든 샘플이 정확히 한 번


def test_sampler_is_reproducible_for_a_fixed_seed():
    lengths = [50, 400, 1200, 90, 2000, 300] * 10
    def run():
        g = torch.Generator().manual_seed(7)
        return list(LengthGroupedBatchSampler(lengths, 3072, 8, 32, g))
    assert run() == run()
