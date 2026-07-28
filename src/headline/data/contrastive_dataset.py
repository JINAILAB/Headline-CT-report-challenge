from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .. import paths
from ..labels import PRESENCE_KEYS
from ..models.prompts import chunk_findings, split_report
from ..studies import iter_studies, load_all_reports

Item = Tuple[str, str, str, str]   # (pe_path, text_path, study_id, patient_id)

# 시간 변화/비시각 문장 → local loss에서 제외
_NONGROUND_RE = re.compile(
    r"\b(compared?|comparison|prior|previous|interval|unchanged|stable|"
    r"progress\w*|regress\w*|increas\w*|decreas\w*|resolv\w*|re-?evaluat\w*|"
    r"follow[- ]?up|earlier|new since|since the (prior|previous|last)|"
    r"clinical|history|indication|technique|recommend\w*|suggest\w*|correlat\w*|"
    r"reconstruct\w*|\[date\])\b", re.I)


def load_presence_labels(splits: Sequence[str] = paths.SPLITS,
                         labels_root: Optional[Path] = None) -> Dict[str, torch.Tensor]:
    """{study_id: (20,) float 0/1}."""
    root = Path(labels_root or paths.CLASSIFIER_LABELS)
    out: Dict[str, torch.Tensor] = {}
    for sp in splits:
        p = root / f"{sp}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            pres = r.get("presence", {})
            out[r["study_id"]] = torch.tensor([float(pres.get(k, 0)) for k in PRESENCE_KEYS])
    return out


def groundable_mask(report: str, n_expected: int) -> List[bool]:
    """chunk_findings와 동일한 라인 분할 → 라인별 groundable 여부. lines_4096과 인덱스를 맞춘다."""
    find, _ = split_report(report)
    lines = chunk_findings(find or report, 16) or [(find or report).strip()[:300]]
    mask = [_NONGROUND_RE.search(s) is None for s in lines]
    if len(mask) != n_expected or not any(mask):
        return [True] * n_expected        # 정렬이 어긋나거나 전부 비시각이면 안전하게 전부 유지
    return mask


def build_items(splits: Sequence[str] = paths.SPLITS, limit: int = 0,
                pe_root: Optional[Path] = None, text_root: Optional[Path] = None,
                labels: Optional[Dict[str, torch.Tensor]] = None) -> List[Item]:
    """pe 캐시 · 텍스트 캐시 · presence 라벨이 **모두** 있는 스터디만."""
    pe_root = Path(pe_root or paths.PE_CACHE)
    text_root = Path(text_root or paths.TEXT_CACHE)
    have_label = set((labels if labels is not None else load_presence_labels(splits)).keys())
    items: List[Item] = []
    for sp in splits:
        for st in iter_studies(sp):
            pe = pe_root / sp / f"{st.study_id}.pt"
            tp = text_root / sp / f"{st.study_id}.pt"
            if not (pe.exists() and tp.exists() and st.study_id in have_label):
                continue
            items.append((str(pe), str(tp), st.study_id, st.patient_id))
            if limit and len(items) >= limit:
                return items
    return items


class ContrastiveDataset(Dataset):
    def __init__(self, items: Sequence[Item], mask_temporal: bool = True,
                 splits: Sequence[str] = paths.SPLITS,
                 labels: Optional[Dict[str, torch.Tensor]] = None):
        self.items = list(items)
        self.mask_temporal = mask_temporal
        self.reports = load_all_reports(splits) if mask_temporal else {}
        self.labels = labels if labels is not None else load_presence_labels(splits)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        pe_path, text_path, sid, _ = self.items[i]
        pe = torch.load(pe_path, weights_only=False)["pe"].float()        # (384,32,32,32)
        t = torch.load(text_path, weights_only=False)
        lines = t["lines_4096"].float()                                   # (C,4096)
        if self.mask_temporal and sid in self.reports:
            gm = torch.tensor(groundable_mask(self.reports[sid], lines.size(0)), dtype=torch.bool)
        else:
            gm = torch.ones(lines.size(0), dtype=torch.bool)
        lab = self.labels.get(sid, torch.zeros(len(PRESENCE_KEYS)))
        return pe, t["find_4096"].float(), lines, gm, lab


def collate(batch):
    pe = torch.stack([b[0] for b in batch])                               # (B,384,32,32,32)
    find = torch.stack([b[1] for b in batch])                             # (B,4096)
    lines = [b[2] for b in batch]
    C = max(l.size(0) for l in lines)
    L = torch.zeros(len(batch), C, 4096)
    M = torch.zeros(len(batch), C, dtype=torch.bool)
    for i, (l, gm) in enumerate(zip(lines, [b[3] for b in batch])):
        L[i, :l.size(0)] = l
        M[i, :l.size(0)] = gm                                             # valid AND groundable
    lab = torch.stack([b[4] for b in batch])                              # (B,20)
    return pe, find, L, M, lab
