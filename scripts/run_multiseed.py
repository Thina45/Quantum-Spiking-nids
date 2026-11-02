#!/usr/bin/env python3
"""Run QSNN-NIDS + all baselines at additional seeds, on one or both datasets.

Seed 42 (artifacts/qsnn_unsw_v2, artifacts/qsnn_cicids2017_v1) is never
touched by this script. Each new seed gets its own run directory
(artifacts/qsnn_<dataset>_seed<N>/) and its own freshly-built data split,
since --seed reseeds data.seed too (see qsnn/data/preprocess.py) -- this
script does NOT pass --bundle, so every seed gets its own train/val/test
partition rather than reusing the seed-42 split.

Idempotent / resumable: a seed+dataset combo already holding results.json
is skipped. A combo with a last.pt but no results.json is resumed with
--resume instead of restarted.

Usage:
    python scripts/run_multiseed.py --datasets unsw_nb15 cicids2017 \
        --seeds 7 13 99 2024
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIG = {
    "unsw_nb15": "configs/unsw_nb15.yaml",
    "cicids2017": "configs/cicids2017.yaml",
}
DATASET_RUNPREFIX = {
    "unsw_nb15": "qsnn_unsw_seed",
    "cicids2017": "qsnn_cicids2017_seed",
}


def run_one(dataset: str, seed: int, log_dir: str) -> None:
    config = DATASET_CONFIG[dataset]
    run_name = f"{DATASET_RUNPREFIX[dataset]}{seed}"
    out_dir = os.path.join(REPO_ROOT, "artifacts", run_name)
    results_path = os.path.join(out_dir, "results.json")
    last_ckpt = os.path.join(out_dir, "last.pt")

    if os.path.exists(results_path):
        print(f"[skip] {run_name} already has results.json")
        return

    cmd = [
        sys.executable, "-u", "scripts/train.py",
        "--config", config,
        "--seed", str(seed),
        "--run-name", run_name,
        # QSVC dropped for repeat seeds: ~2-2.65h/run, not the paper's novel
        # contribution. Already have it at seed 42; report that as single-seed.
        "--baselines", "rf", "svm", "mlp", "qcnn",
        # deliberately no --bundle: seed must drive its own data split
    ]
    if os.path.exists(last_ckpt):
        cmd += ["--resume", last_ckpt]
        print(f"[resume] {run_name} from {last_ckpt}")
    else:
        print(f"[start] {run_name}")

    log_path = os.path.join(log_dir, f"{run_name}.log")
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a") as fh:
        fh.write(f"\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"[{status}] {run_name} in {secs/60:.1f} min  (log: {log_path})")
    if r.returncode != 0:
        raise SystemExit(f"{run_name} failed -- see {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["unsw_nb15", "cicids2017"],
                    choices=list(DATASET_CONFIG))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--log-dir", default="logs/multiseed")
    a = ap.parse_args()

    for seed in a.seeds:
        for dataset in a.datasets:
            run_one(dataset, seed, a.log_dir)


if __name__ == "__main__":
    main()
