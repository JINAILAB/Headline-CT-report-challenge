from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .. import paths
from ..data.ct_input import load_brain_ct_11window
from ..models.pillar import PillarFeatureExtractor
from ..studies import iter_studies


class _CTDataset(Dataset):
    """아직 캐싱되지 않은 스터디의 11-window CT."""

    def __init__(self, out_root: Path, splits):
        self.items = []
        for sp in splits:
            for st in iter_studies(sp):
                wt = st.recon_path("WT")
                if wt is None or (out_root / sp / f"{st.study_id}.pt").exists():
                    continue
                self.items.append((st.study_id, sp, str(wt)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, sp, wt = self.items[i]
        return sid, sp, load_brain_ct_11window(wt, znorm=False)


def _collate(b):
    return [x[0] for x in b], [x[1] for x in b], torch.stack([x[2] for x in b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.PE_CACHE)
    ap.add_argument("--splits", nargs="+", default=list(paths.SPLITS))
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for sp in a.splits:
        (a.out / sp).mkdir(parents=True, exist_ok=True)

    fe = PillarFeatureExtractor(trainable=False)
    patch_embed = fe.visual.patch_embeds["brain_ct"]

    ds = _CTDataset(a.out, a.splits)
    if a.limit:
        ds.items = ds.items[:a.limit]
    print(f"[pe_cache] 남은 스터디: {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    collate_fn=_collate, prefetch_factor=2, persistent_workers=a.workers > 0)

    t0, n = time.time(), 0
    for sids, sps, ct in dl:
        with torch.no_grad():
            pe = patch_embed(ct.to("cuda", torch.float32))        # (B,384,32,32,32)
        pe = pe.to(torch.float16).cpu()
        for k, sid in enumerate(sids):
            torch.save({"study_id": sid, "split": sps[k], "pe": pe[k].clone()},
                       a.out / sps[k] / f"{sid}.pt")
        n += len(sids)
        if n % 200 < a.bs:
            print(f"  {n} cached ({(time.time()-t0)/max(n,1)*1000:.0f} ms/study)", flush=True)
    print(f"[pe_cache] {n} studies → {a.out} ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
