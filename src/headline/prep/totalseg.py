from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .. import paths
from ..studies import Study, iter_studies

TS_BIN = "TotalSegmentator"
TASKS = [("brain_structures", "brain_regions.nii.gz"),
         ("craniofacial_structures", "bone_craniofacial.nii.gz")]


@dataclass
class Config:
    split: str
    tasks: List[str] = field(default_factory=lambda: [t for t, _ in TASKS])
    gpus: Tuple[int, ...] = (0,)
    workers: int = 8
    stagger: float = 3.0        # 신규 launch 최소 간격(초)
    retries: int = 2
    threads: int = 8            # 프로세스당 스레드 (workers × threads ≈ 코어 수)
    limit: int = 0
    out_root: Path = paths.MASKS


@dataclass
class Job:
    study: Study
    task: str
    out_file: Path
    cmd: List[str]
    attempt: int = 0


def _ts_version() -> str:
    try:
        return subprocess.run([TS_BIN, "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return "unknown"


def build_jobs(cfg: Config) -> Tuple[List[Job], int, int]:
    jobs, skipped = [], 0
    studies = list(iter_studies(cfg.split))
    if cfg.limit:
        studies = studies[:cfg.limit]
    for st in studies:
        wt = st.recon_path("WT")
        if wt is None:
            continue
        for task, fname in TASKS:
            if task not in cfg.tasks:
                continue
            out_file = cfg.out_root / st.split / st.patient_id / st.study_id / fname
            if out_file.exists() and out_file.stat().st_size > 0:
                skipped += 1
                continue
            # -nr/-ns 1: 리샘플/저장 스레드를 1로 묶어 프로세스당 스레드·fork 폭증을 막는다
            # (컨테이너 cgroup pids.max 초과로 BlockingIOError가 난다).
            cmd = [TS_BIN, "-i", str(wt), "-o", str(out_file), "-ml", "-ta", task,
                   "--device", "gpu", "-nr", "1", "-ns", "1"]
            jobs.append(Job(st, task, out_file, cmd))
    return jobs, skipped, len(studies)


def _write_meta(cfg: Config, study: Study, ok: List[str], fail: List[str]) -> None:
    """provenance. 원래 파이프라인은 병렬 실행 경로에서 이걸 빠뜨려 기록이 남지 않았다."""
    meta_file = cfg.out_root / study.split / study.patient_id / study.study_id / "_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    meta.update(study_id=study.study_id, patient_id=study.patient_id, split=study.split,
                reference_grid="WT")
    meta.setdefault("sources", {})["totalsegmentator"] = {
        "tasks": {t: f for t, f in TASKS if t in cfg.tasks},
        "version": _ts_version(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok": ok, "fail": fail,
    }
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _env(cfg: Config, gpu: int) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OPENCV_NUM_THREADS"):
        env[k] = str(cfg.threads)
    return env


def run(cfg: Config) -> Tuple[int, int]:
    jobs, skipped, total = build_jobs(cfg)
    print(f"[totalseg] split={cfg.split} studies={total} todo={len(jobs)} skip={skipped} "
          f"workers={cfg.workers} gpus={cfg.gpus} ts={_ts_version()}", flush=True)
    pending, running = list(jobs), []
    done = failed = 0
    per_study_ok, per_study_fail = {}, {}
    n_todo, last_launch, gpu_rr, t0 = len(pending), 0.0, 0, time.time()

    def launch(job: Job) -> bool:
        nonlocal gpu_rr, last_launch
        gpu = cfg.gpus[gpu_rr % len(cfg.gpus)]
        job.out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.Popen(job.cmd, env=_env(cfg, gpu),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (BlockingIOError, OSError) as e:      # fork EAGAIN / pids.max 초과 → 백오프
            sys.stderr.write(f"  [backoff] {type(e).__name__}: {e} (running={len(running)})\n")
            time.sleep(5)
            return False
        gpu_rr += 1
        running.append((proc, job))
        last_launch = time.time()
        return True

    while pending or running:
        while pending and len(running) < cfg.workers and (time.time() - last_launch) >= cfg.stagger:
            if launch(pending[0]):
                pending.pop(0)
            else:
                break
        still = []
        for proc, job in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, job))
                continue
            ok = rc == 0 and job.out_file.exists() and job.out_file.stat().st_size > 0
            key = (job.study.split, job.study.patient_id, job.study.study_id)
            if ok:
                done += 1
                per_study_ok.setdefault(key, (job.study, []))[1].append(job.task)
            else:
                job.attempt += 1
                if job.attempt <= cfg.retries:
                    pending.append(job)              # 맨 뒤로 → 자연스러운 backoff
                else:
                    failed += 1
                    per_study_fail.setdefault(key, (job.study, []))[1].append(job.task)
                    err = (proc.stderr.read() or b"")[-300:].decode("utf-8", "ignore")
                    sys.stderr.write(f"  [FAIL x{job.attempt}] {job.study.study_id} {job.task}\n{err}\n")
            if (done + failed) % 50 == 0 and (done + failed):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(pending) + len(still)) / rate / 3600 if rate else float("inf")
                print(f"  {done}/{n_todo} done, {failed} fail, {len(running)} run, "
                      f"{rate:.3f} job/s, ETA {eta:.1f}h", flush=True)
        running = still
        time.sleep(0.5)

    for key in set(per_study_ok) | set(per_study_fail):
        study = (per_study_ok.get(key) or per_study_fail[key])[0]
        _write_meta(cfg, study, per_study_ok.get(key, (None, []))[1],
                    per_study_fail.get(key, (None, []))[1])
    el = time.time() - t0
    print(f"[totalseg done] split={cfg.split} done={done} failed={failed} in {el/3600:.2f}h", flush=True)
    return done, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(paths.SPLITS) + ["validation"])
    ap.add_argument("--tasks", nargs="+", default=[t for t, _ in TASKS])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stagger", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_root", type=Path, default=paths.MASKS)
    a = ap.parse_args()
    run(Config(split=a.split, tasks=a.tasks, gpus=tuple(int(x) for x in a.gpus.split(",")),
               workers=a.workers, stagger=a.stagger, retries=a.retries, threads=a.threads,
               limit=a.limit, out_root=a.out_root))


if __name__ == "__main__":
    main()
