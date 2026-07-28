from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from .. import paths
from ..studies import load_reports
from .report_schema import REPORT_SCHEMA, build_messages
from .report_schema_detailed import DETAILED_SCHEMA, build_messages_detailed


def extract_last_json(text: str):
    """마지막으로 완전히 파싱되는 JSON 객체를 반환.

    gpt-oss는 reasoning과 최종 답을 한 문자열로 합쳐 내보내므로(harmony 채널), reasoning 안의
    stray '{'를 피하려면 앞에서부터 훑어 **마지막** 성공을 취해야 한다.
    """
    dec = json.JSONDecoder()
    best, idx = None, 0
    while idx < len(text):
        i = text.find("{", idx)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                best = obj
            idx = i + max(end - i, 1)
        except json.JSONDecodeError:
            idx = i + 1
    return best


def _load_rows(a) -> List[Tuple[str, str]]:
    if a.source == "reports":
        return list(load_reports(a.split).items())
    with open(a.gen_csv, newline="") as fh:
        return [(r["study_id"], r["report"]) for r in csv.DictReader(fh)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.GPT_OSS))
    ap.add_argument("--source", default="reports", choices=["reports", "csv"])
    ap.add_argument("--split", choices=list(paths.SPLITS), help="--source reports일 때 필요")
    ap.add_argument("--gen_csv", help="--source csv일 때 필요")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--schema", default="core", choices=["core", "detailed"])
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=0, help="0이면 스키마별 기본값")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--reasoning", default="low", help="gpt-oss reasoning effort")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    assert a.source != "reports" or a.split, "--source reports에는 --split이 필요하다"
    assert a.source != "csv" or a.gen_csv, "--source csv에는 --gen_csv가 필요하다"

    detailed = a.schema == "detailed"
    schema = DETAILED_SCHEMA if detailed else REPORT_SCHEMA
    build = build_messages_detailed if detailed else build_messages
    max_model_len = a.max_model_len or (16384 if detailed else 8192)
    max_tokens = a.max_tokens or (7000 if detailed else 4000)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(a)[a.shard::a.nshards]     # 샤딩은 전체 목록 기준이라 재시작해도 겹치지 않는다
    if a.limit:
        rows = rows[:a.limit]
    todo = [(sid, txt) for sid, txt in rows if not (a.out_dir / f"{sid}.json").exists()]
    print(f"[extract] total={len(rows)} todo={len(todo)} schema={a.schema} → {a.out_dir}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True)
    # max_tokens는 넉넉히: reasoning이 길면 JSON 전에 잘린다.
    # repetition_penalty는 guided decoding에서 생기는 퇴행성 반복을 추가로 억제한다.
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.05,
                        structured_outputs=StructuredOutputsParams(json=schema))
    chat_kwargs = {"reasoning_effort": a.reasoning} if "oss" in Path(a.model).name else {}

    for s in range(0, len(todo), a.chunk):
        batch = todo[s:s + a.chunk]
        msgs = [build(txt) for _, txt in batch]
        try:
            outs = llm.chat(msgs, sp, chat_template_kwargs=chat_kwargs) if chat_kwargs \
                else llm.chat(msgs, sp)
        except TypeError:                          # 구버전 vLLM은 chat_template_kwargs 미지원
            outs = llm.chat(msgs, sp)
        for (sid, _), o in zip(batch, outs):
            text = o.outputs[0].text
            data = extract_last_json(text) or {"_parse_error": text[:200]}
            data["_study_id"] = sid
            (a.out_dir / f"{sid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"  {min(s + a.chunk, len(todo))}/{len(todo)} done", flush=True)
    print("[extract] 완료", flush=True)


if __name__ == "__main__":
    main()
