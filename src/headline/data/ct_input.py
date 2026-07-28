from __future__ import annotations

import torch
import torch.nn.functional as F

import rve

TARGET_DHW = (128, 256, 256)   # (D,H,W) — Pillar image_size [256,256,128]=(H,W,D)
MEAN, STD = 0.289, 0.198
CT_WINDOWS = list(rve.get_available_windows("CT"))   # 11개, 고정 순서
N_WINDOWS = len(CT_WINDOWS)


def load_brain_ct_11window(nii_path: str, device: str = "cpu",
                           znorm: bool = False) -> torch.Tensor:
    """nii → (11, 128, 256, 256) float32.

    znorm=True면 (x-MEAN)/STD, False면 window [0,1] 그대로. 기본값은 재정렬 encoder 기준인 False다.
    """
    vol = rve.load_nifti(nii_path)                  # (H,W,D) raw HU
    if not torch.is_tensor(vol):
        vol = torch.as_tensor(vol)
    vol = vol.float().permute(2, 0, 1).contiguous()  # (D,H,W)
    # HU 의미를 보존하기 위해 windowing 전에 resize 한다.
    vol = F.interpolate(vol[None, None], size=TARGET_DHW, mode="trilinear",
                        align_corners=False)[0, 0]   # (128,256,256)
    chans = [torch.as_tensor(rve.apply_windowing(vol, w, "CT")).float() for w in CT_WINDOWS]
    win = torch.stack(chans, dim=0)                  # (11,128,256,256), [0,1]
    assert win.shape == (N_WINDOWS, *TARGET_DHW), win.shape
    if znorm:
        win = (win - MEAN) / STD
    return win.to(device).float()
