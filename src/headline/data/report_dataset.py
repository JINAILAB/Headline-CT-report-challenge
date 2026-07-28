from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .. import paths
from ..models.prompts import (GEN_INSTRUCTION, STAGE1_INSTRUCTION, STAGE1_SYSTEM,
                              wrap_post, wrap_pre, wrap_target)
from ..splits import load_val_patients, patient_holdout
from ..studies import load_all_reports, study_patient_map


def _tokenizer_fn(tokenizer) -> Callable[[str], torch.Tensor]:
    def t(text: str) -> torch.Tensor:
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return t


def _load_cached(pt_path: Path) -> dict:
    d = torch.load(pt_path, weights_only=False)
    # grids는 fp16 그대로 둔다(host RAM 절반; 모델이 GPU에서 float로 올린다).
    return dict(grids=list(d["grids"]),
                global_embed=d["global_embed"].float(),
                region_masks=d["region_masks32"].float())


class CachedReportDataset(Dataset):
    def __init__(self, cache_root: Path, splits: Sequence[str] = paths.SPLITS,
                 val: bool = False, limit: int = 0,
                 val_patients: Optional[set] = None, raw_root: Optional[Path] = None):
        self.reports = load_all_reports(splits, raw_root)
        self.pid = study_patient_map(splits, raw_root)
        items: List[Tuple[str, str, Path]] = []
        for sp in splits:
            for pt in sorted((Path(cache_root) / sp).glob("*.pt")):
                if pt.stem in self.reports:
                    items.append((sp, pt.stem, pt))
        if limit:                       # 스모크 모드: 몇십 건으로는 val split이 성립하지 않는다
            self.items = items[:limit]
        else:
            vp = val_patients if val_patients is not None else load_val_patients()
            self.items = patient_holdout(items, lambda it: self.pid.get(it[1], it[1]), val, vp)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        _, sid, pt = self.items[i]
        out = _load_cached(pt)
        out.update(pre_text=wrap_pre("", STAGE1_SYSTEM),
                   post_text=wrap_post(STAGE1_INSTRUCTION),
                   target_text=wrap_target(self.reports[sid]))
        return out

    def report_texts(self) -> List[str]:
        return [self.reports[sid] for (_, sid, _) in self.items]

    def study_ids(self) -> List[str]:
        return [sid for (_, sid, _) in self.items]


class AugmentedReportDataset(Dataset):

    def __init__(self, cache_root: Path, splits: Sequence[str] = paths.SPLITS,
                 aug_root: Optional[Path] = None, limit: int = 0,
                 val_patients: Optional[set] = None, raw_root: Optional[Path] = None):
        aug_root = Path(aug_root or paths.AUGMENT)
        self.pid = study_patient_map(splits, raw_root)
        cache_path: Dict[Tuple[str, str], Path] = {
            (sp, pt.stem): pt for sp in splits for pt in (Path(cache_root) / sp).glob("*.pt")}
        items: List[Tuple[str, Path, dict]] = []
        for sp in splits:
            jf = aug_root / f"{sp}.jsonl"
            if not jf.exists():
                continue
            for line in open(jf):
                inst = json.loads(line)
                pt = cache_path.get((sp, inst["study_id"]))
                if pt is not None:
                    items.append((inst["study_id"], pt, inst))
        if limit:
            self.items = items[:limit]
        else:   # 스터디 1건이 인스턴스 여러 건이므로 val 환자 인스턴스를 전부 제거
            vp = val_patients if val_patients is not None else load_val_patients()
            self.items = [it for it in items if self.pid.get(it[0], it[0]) not in vp]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        _, pt, inst = self.items[i]
        out = _load_cached(pt)
        out.update(pre_text=wrap_pre(inst.get("user_context", "")),
                   post_text=wrap_post(inst["instruction"]),
                   target_text=wrap_target(inst["target"]))
        return out

    def length_texts(self) -> List[str]:
        """길이 추정용 전체 텍스트. target만 세면 user_context가 긴 태스크를 과소평가해 OOM이 난다."""
        return [wrap_pre(i.get("user_context", "")) + wrap_post(i["instruction"]) + i["target"]
                for (_, _, i) in self.items]

    def task_types(self) -> List[str]:
        return [it[2]["task_type"] for it in self.items]


def make_collate(tokenizer) -> Callable[[List[dict]], dict]:
    t = _tokenizer_fn(tokenizer)

    def collate(batch: List[dict]) -> dict:
        grids = [torch.stack([b["grids"][s] for b in batch]) for s in range(3)]  # [(B,D,H,W,384)]
        return dict(
            cached=(grids, torch.stack([b["global_embed"] for b in batch])),      # (B,1152)
            region_masks=torch.stack([b["region_masks"] for b in batch]),         # (B,18,32,32,32)
            pre_ids=[t(b["pre_text"]) for b in batch],
            post_ids=[t(b["post_text"]) for b in batch],
            report_ids=[t(b["target_text"]) for b in batch],
        )
    return collate


def make_eval_collate(tokenizer) -> Callable[[List[dict]], dict]:
    """평가 생성용"""
    t = _tokenizer_fn(tokenizer)
    pre, post = t(wrap_pre("")), t(wrap_post(GEN_INSTRUCTION))

    def collate(batch: List[dict]) -> dict:
        grids = [torch.stack([b["grids"][s] for b in batch]) for s in range(3)]
        return dict(
            cached=(grids, torch.stack([b["global_embed"] for b in batch])),
            region_masks=torch.stack([b["region_masks"] for b in batch]),
            pre_ids=[pre] * len(batch),
            post_ids=[post] * len(batch),
            report_ids=[t(b["target_text"]) for b in batch],
        )
    return collate
