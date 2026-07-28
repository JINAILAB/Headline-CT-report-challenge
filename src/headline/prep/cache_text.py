from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from .. import paths
from ..models.prompts import chunk_findings, split_report
from ..studies import load_reports


class QwenEmbedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: List[str], bs: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), bs):
            inp = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            h = self.model(**inp).last_hidden_state                     # (B,T,4096)
            m = inp["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)             # mean-pool
            out.append(F.normalize(emb, dim=1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 4096)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.QWEN_EMBED))
    ap.add_argument("--out", type=Path, default=paths.TEXT_CACHE)
    ap.add_argument("--splits", nargs="+", default=list(paths.SPLITS))
    ap.add_argument("--max_lines", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    emb = QwenEmbedder(a.model)
    t0, n = time.time(), 0
    for sp in a.splits:
        out_dir = a.out / sp
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid, report in load_reports(sp).items():
            pt = out_dir / f"{sid}.pt"
            if pt.exists():                                             # 재개
                n += 1
                continue
            find, _ = split_report(report)
            find = find or report
            lines = chunk_findings(find, a.max_lines) or [find.strip()[:300]]
            e = emb.embed([find.strip()] + lines)                        # (1+C, 4096)
            torch.save({"study_id": sid, "find_4096": e[0].clone(),
                        "lines_4096": e[1:].clone(), "n_lines": len(lines)}, pt)
            n += 1
            if n % 500 == 0:
                print(f"  {n} cached ({(time.time()-t0)/n*1000:.0f} ms/study)", flush=True)
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    print(f"[cache_text] {n} studies → {a.out} ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
