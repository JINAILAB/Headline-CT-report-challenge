from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .. import paths
from ..models.aligner import load_encoder_ckpt
from ..models.pillar import PillarFeatureExtractor
from ..studies import iter_studies
from .ct_input import load_brain_ct_11window
from .masks import load_region_masks32

FINE_GRID = (32, 32, 32)


@dataclass
class CacheArgs:
    split: str
    source: str = "raw"                     # raw | pe_cache
    shard: int = 0
    nshards: int = 1
    enc_ckpt: Optional[Path] = None
    out_root: Path = paths.PILLAR_CACHE
    pe_root: Path = paths.PE_CACHE
    limit: int = 0


def build_extractor(enc_ckpt: Optional[Path]) -> tuple:
    """(PillarFeatureExtractor, znorm) — encoder 체크포인트가 znorm을 결정한다."""
    fe = PillarFeatureExtractor(device="cuda")
    if enc_ckpt is None:
        return fe, True                      # 원본 frozen Pillar-0 캐시는 znorm=True
    sd, znorm = load_encoder_ckpt(Path(enc_ckpt))
    missing, unexpected = fe.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    fe.model.eval()
    print(f"[enc] {enc_ckpt}: {len(sd)} tensors (missing {len(missing)}) znorm={znorm}", flush=True)
    return fe, znorm


def run(args: CacheArgs) -> None:
    fe, znorm = build_extractor(args.enc_ckpt)
    studies = [s for i, s in enumerate(iter_studies(args.split)) if i % args.nshards == args.shard]
    if args.limit:
        studies = studies[:args.limit]
    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    n = ok = 0
    for st in studies:
        n += 1
        out = out_dir / f"{st.study_id}.pt"
        if out.exists():                                     # resume: 이미 만든 건 건너뛴다
            ok += 1
            continue
        try:
            if args.source == "pe_cache":
                pe_path = Path(args.pe_root) / args.split / f"{st.study_id}.pt"
                if not pe_path.exists():
                    continue
                pe = torch.load(pe_path, weights_only=False)["pe"].float().unsqueeze(0)
                feats = fe.forward_cached(pe.cuda())
            else:
                src = st.recon_path("WT")
                if src is None:
                    continue
                x = load_brain_ct_11window(str(src), znorm=znorm)     # (11,128,256,256)
                feats = fe(x.unsqueeze(0).cuda())
            rec = {
                "grids": [g[0].half().cpu() for g in feats.grids],    # [(D,H,W,384) fp16]
                "global_embed": feats.global_embed[0].half().cpu(),   # (1152,)
                "region_masks32": load_region_masks32(st.mask_dir, FINE_GRID),   # (18,32,32,32) uint8
            }
            tmp = out.with_suffix(".tmp")
            torch.save(rec, tmp)
            tmp.replace(out)                                  # 원자적 저장(중단돼도 손상 없음)
            ok += 1
        except Exception as e:                                # noqa: BLE001 — 한 건 실패로 전체를 멈추지 않는다
            sys.stderr.write(f"[FAIL] {st.study_id}: {type(e).__name__} {e}\n")
        if n % 100 == 0:
            print(f"  [{args.split} sh{args.shard}] {n}/{len(studies)} (ok {ok})", flush=True)
    print(f"[done {args.split} sh{args.shard}] {ok}/{len(studies)} → {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pillar feature 캐시 생성")
    ap.add_argument("--split", required=True, choices=list(paths.SPLITS))
    ap.add_argument("--source", default="raw", choices=["raw", "pe_cache"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--enc_ckpt", type=Path, default=None,
                    help="재정렬 encoder 체크포인트(headline.train.realign --export 산출). "
                         "생략하면 원본 Pillar-0.")
    ap.add_argument("--out_root", type=Path, default=paths.PILLAR_CACHE)
    ap.add_argument("--pe_root", type=Path, default=paths.PE_CACHE)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(CacheArgs(split=a.split, source=a.source, shard=a.shard, nshards=a.nshards,
                  enc_ckpt=a.enc_ckpt, out_root=a.out_root, pe_root=a.pe_root, limit=a.limit))


if __name__ == "__main__":
    main()
