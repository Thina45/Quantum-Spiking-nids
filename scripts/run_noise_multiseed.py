#!/usr/bin/env python3
"""Run the 6-level depolarizing noise sweep at multiple seeds, on one or both
datasets, reusing the existing stratified-2000-sample-subsample approach.

Requires that artifacts/<run_name>/final.pt already exists for every
(dataset, seed) pair -- i.e. run_multiseed.py must have trained that seed's
QSNN first. Seed 42's noise sweep (artifacts/qsnn_unsw_v2/noise_sweep.json,
artifacts/qsnn_cicids2017_v1/noise_sweep.json) already exists and is never
re-run or overwritten by this script.

Each seed gets its own subsample of ITS OWN test set (not seed 42's), since
every seed has its own data split -- the subsample is built fresh from that
seed's run bundle via a quick rebuild of the processed data at that seed.

Usage:
    python scripts/run_noise_multiseed.py --datasets unsw_nb15 cicids2017 \
        --seeds 7 13
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DATASET_CONFIG = {
    "unsw_nb15": "configs/unsw_nb15.yaml",
    "cicids2017": "configs/cicids2017.yaml",
}
DATASET_RUNPREFIX = {
    "unsw_nb15": "qsnn_unsw_seed",
    "cicids2017": "qsnn_cicids2017_seed",
}
LEVELS = ["0", "0.005", "0.01", "0.02", "0.05", "0.1"]


def build_seed_subsample(dataset: str, seed: int, n: int = 2000) -> str:
    """Builds data/processed/<dataset>_seed<seed>_noisesub.npz from a fresh,
    seed-specific split (NOT the seed-42 bundle), then subsamples its test
    set. Returns the subsample path."""
    from qsnn.config import Config
    from qsnn.data.preprocess import build_dataset
    from qsnn.engine import set_seed
    import json
    import numpy as np

    config = DATASET_CONFIG[dataset]
    cfg = Config.from_yaml(config).override({"train.seed": seed, "data.seed": seed})
    set_seed(seed)
    bundle = build_dataset(cfg, verbose=False)

    full_path = f"data/processed/{dataset}_seed{seed}.npz"
    sub_path = f"data/processed/{dataset}_seed{seed}_noisesub.npz"

    y = bundle.y_test
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    frac = n / len(y)
    idx_parts = []
    for c, cnt in zip(classes, counts):
        take = max(1, int(round(cnt * frac)))
        cls_idx = np.where(y == c)[0]
        idx_parts.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)

    os.makedirs("data/processed", exist_ok=True)
    np.savez(
        sub_path,
        X_train=bundle.X_train, y_train=bundle.y_train,
        X_val=bundle.X_val, y_val=bundle.y_val,
        X_test=bundle.X_test[idx], y_test=bundle.y_test[idx],
        feature_names=np.array(bundle.feature_names, dtype=object),
        class_names=np.array(bundle.class_names, dtype=object),
        class_weights=bundle.class_weights,
        meta=np.array([json.dumps(bundle.meta)], dtype=object),
    )
    print(f"[io] seed {seed} {dataset}: subsampled test {len(y)} -> {len(idx)} -> {sub_path}")
    return sub_path


def run_one(dataset: str, seed: int, log_dir: str) -> None:
    config = DATASET_CONFIG[dataset]
    run_name = f"{DATASET_RUNPREFIX[dataset]}{seed}"
    out_dir = os.path.join(REPO_ROOT, "artifacts", run_name)
    ckpt = os.path.join(out_dir, "final.pt")
    out_json = os.path.join(out_dir, "noise_sweep.json")

    if os.path.exists(out_json):
        print(f"[skip] {run_name} already has noise_sweep.json")
        return
    if not os.path.exists(ckpt):
        print(f"[wait] {run_name}: no final.pt yet -- run_multiseed.py must "
              f"finish this (dataset, seed) first. Skipping for now.")
        return

    sub_path = build_seed_subsample(dataset, seed)

    cmd = [
        sys.executable, "-u", "scripts/eval_noise.py",
        "--config", config,
        "--checkpoint", ckpt,
        "--bundle", sub_path,
        "--levels", *LEVELS,
        "--out", out_json,
    ]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_noise.log")
    print(f"[start] noise sweep {run_name}")
    with open(log_path, "a") as fh:
        r = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"[{status}] noise sweep {run_name} (log: {log_path})")
    if r.returncode != 0:
        raise SystemExit(f"noise sweep {run_name} failed -- see {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["unsw_nb15", "cicids2017"],
                    choices=list(DATASET_CONFIG))
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="extra seeds to noise-sweep (seed 42 already done, do not pass it)")
    ap.add_argument("--log-dir", default="logs/multiseed")
    a = ap.parse_args()

    for seed in a.seeds:
        for dataset in a.datasets:
            run_one(dataset, seed, a.log_dir)


if __name__ == "__main__":
    main()
