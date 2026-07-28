from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):

    def __init__(self, lengths: List[int], max_tokens: int, max_bs: int = 32,
                 megabatch_size: int = 256, generator: Optional[torch.Generator] = None):
        self.lengths = [int(x) for x in lengths]
        self.max_tokens = int(max_tokens)
        self.max_bs = int(max_bs)
        self.megabatch_size = int(megabatch_size)
        self.generator = generator
        assert self.lengths and min(self.lengths) >= 1
        self._est_len = self._estimate_len()

    def _pack(self, mb: List[int]) -> List[List[int]]:
        # 길이 내림차순 정렬 → cur[0]이 항상 batch 내 최장. 탐욕적으로 예산을 채운다.
        mb = sorted(mb, key=lambda i: self.lengths[i], reverse=True)
        out: List[List[int]] = []
        cur: List[int] = []
        for i in mb:
            if cur and ((len(cur) + 1) * self.lengths[cur[0]] > self.max_tokens
                        or len(cur) >= self.max_bs):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def _estimate_len(self) -> int:
        # 결정적 추정치(len(dataloader)용): 전체 내림차순 탐욕 패킹의 batch 수.
        lens = sorted(self.lengths, reverse=True)
        n = i = 0
        while i < len(lens):
            bs = max(1, min(self.max_bs, self.max_tokens // lens[i]))
            i += bs
            n += 1
        return n

    def __len__(self) -> int:
        return self._est_len

    def __iter__(self):
        g = self.generator
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        batches: List[List[int]] = []
        for s in range(0, len(idx), self.megabatch_size):
            batches.extend(self._pack(idx[s:s + self.megabatch_size]))
        order = torch.randperm(len(batches), generator=g).tolist()
        return iter([batches[o] for o in order])
