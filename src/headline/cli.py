from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Type, TypeVar

T = TypeVar("T")


def _to_argv(data: dict) -> List[str]:
    argv: List[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            if value:                       # store_true 계열은 값 없이 플래그로
                argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            argv.append(f"--{key}")
            argv.extend(str(v) for v in value)
        elif value is not None:
            argv.append(f"--{key}={value}")
    return argv


def parse_config(cls: Type[T], argv: Sequence[str] | None = None) -> T:
    from transformers import HfArgumentParser

    argv = list(sys.argv[1:] if argv is None else argv)
    base: List[str] = []
    if argv and argv[0] == "--config":
        base = _to_argv(json.loads(Path(argv[1]).read_text()))
        argv = argv[2:]
    elif len(argv) == 1 and argv[0].endswith(".json"):      # 위치 인자로 준 경우
        base, argv = _to_argv(json.loads(Path(argv[0]).read_text())), []
    (cfg,) = HfArgumentParser(cls).parse_args_into_dataclasses(args=base + argv)
    return cfg
