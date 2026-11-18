#!/usr/bin/env python3
"""Classical SNN ablation: QSNN-NIDS with the quantum circuit replaced by a
classical current source (qsnn/models/classical_snn.py), run on both
datasets x 5 seeds, matching the exact preprocessing/hyperparameters/seed
handling of the existing QSNN-NIDS runs (no --bundle, so each seed rebuilds
its own fresh train/val/test split via build_dataset(cfg), same as
scripts/run_multiseed.py did for the QSNN-NIDS results).

Resumable: a run with results.json already present is skipped; a run with
last.pt but no results.json resumes from the last checkpoint (engine.train's
existing resume_path mechanism, unmodified).

Outputs (all new; no existing file is read for writing, nothing in
artifacts/ or configs/ is touched):
    results/revision/classical_snn/runs/<run_name>/   (checkpoints, history, results.json)
    results/revision/classical_snn/per_seed.csv
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset
from qsnn.engine import evaluate, make_loaders, resolve_device, set_seed, train
from qsnn.models.classical_snn import build_classical_snn_model
from qsnn.utils import banner, save_json

OUT_ROOT = "results/revision/classical_snn"
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
PER_SEED_CSV = os.path.join(OUT_ROOT, "per_seed.csv")

DATASETS = {
    "unsw_nb15": "configs/unsw_nb15.yaml",
    "cicids2017": "configs/cicids2017.yaml",
}
SEEDS = [42, 7, 13, 99, 2024]

CSV_FIELDS = [
    "dataset", "seed", "status", "epochs_run", "wall_seconds",
    "accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "mcc",
    "spike_rate", "latency_ms_per_sample",
    "current_source_params", "other_classical_params", "total_params",
    "torch_version", "python_version", "cpu", "n_cpu",
]


def env_info() -> dict:
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "n_cpu": os.cpu_count(),
    }


def append_csv_row(row: dict) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    new_file = not os.path.exists(PER_SEED_CSV)
    with open(PER_SEED_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_one(dataset: str, seed: int) -> None:
    run_name = f"classical_snn_{dataset}_seed{seed}"
    out_dir = os.path.join(RUNS_DIR, run_name)
    results_path = os.path.join(out_dir, "results.json")
    last_ckpt = os.path.join(out_dir, "last.pt")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(results_path):
        print(f"[skip] {run_name} already has results.json")
        return

    print(banner(f"CLASSICAL SNN ABLATION  |  {dataset}  |  seed={seed}"))
    env = env_info()
    t0 = time.time()

    try:
        cfg = Config.from_yaml(DATASETS[dataset]).override({
            "train.seed": seed, "data.seed": seed,
            "train.run_name": run_name, "train.out_dir": RUNS_DIR,
        })
        set_seed(cfg.train.seed)

        # no --bundle: rebuild the split fresh from this seed, exactly as
        # scripts/run_multiseed.py did for the QSNN-NIDS seed runs (no
        # persisted full per-seed bundle file exists for those runs either).
        bundle = build_dataset(cfg)
        cfg.model.n_classes = bundle.n_classes

        model = build_classical_snn_model(cfg, bundle.n_features)
        param_counts = model.param_counts()
        print(f"  params: {param_counts}")

        loaders = make_loaders(bundle, cfg)
        resume_path = last_ckpt if os.path.exists(last_ckpt) else None
        model, hist = train(model, loaders, cfg, bundle.class_weights,
                            bundle.n_classes, resume_path=resume_path)

        device = resolve_device(cfg.train.device)
        m, y_true, y_pred, y_score = evaluate(model, loaders["test"], device, bundle.n_classes)

        # latency, measured the same way as QSNN's benchmark_latency (engine.py
        # full_evaluation): a warmed-up timed forward pass on <=256 test flows.
        model.eval()
        xb = torch.as_tensor(bundle.X_test[:min(256, len(bundle.X_test))],
                             dtype=torch.float32).to(device)
        with torch.no_grad():
            model(xb[:1])  # warm up
            best_ms = None
            for _ in range(3):
                t_lat = time.perf_counter()
                model(xb)
                secs = time.perf_counter() - t_lat
                if best_ms is None or secs < best_ms:
                    best_ms = secs
        latency_ms_per_sample = best_ms * 1e3 / max(len(xb), 1)

        wall_seconds = time.time() - t0
        status = "OK"

        out = {
            "status": status, "dataset": dataset, "seed": seed,
            "epochs_run": len(hist.rows), "wall_seconds": wall_seconds,
            "detection_metrics": m,
            "spike_rate": model.spike_rate,
            "latency_ms_per_sample": latency_ms_per_sample,
            "param_counts": param_counts,
            "env": env,
            "config": cfg.to_dict(),
        }
        save_json(out, results_path)
        cfg.save(os.path.join(out_dir, "config.used.yaml"))

        row = {
            "dataset": dataset, "seed": seed, "status": status,
            "epochs_run": out["epochs_run"], "wall_seconds": round(wall_seconds, 1),
            **{k: m.get(k) for k in
               ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "mcc")},
            "spike_rate": model.spike_rate,
            "latency_ms_per_sample": latency_ms_per_sample,
            **param_counts, **env,
        }
        append_csv_row(row)
        print(f"[OK] {run_name} in {wall_seconds / 60:.1f} min")

    except Exception as exc:
        wall_seconds = time.time() - t0
        status = f"FAILED: {type(exc).__name__}: {exc}"
        print(banner(f"{run_name} FAILED"))
        print(f"  {status}")
        save_json({"status": status, "dataset": dataset, "seed": seed,
                   "wall_seconds": wall_seconds, "env": env},
                  os.path.join(out_dir, "FAILED.json"))
        append_csv_row({"dataset": dataset, "seed": seed, "status": status,
                        "wall_seconds": round(wall_seconds, 1), **env})
        # do not re-raise: record the failure and continue with the remaining
        # seeds/datasets, per "record it as such" rather than aborting the batch.


def main() -> None:
    for dataset in DATASETS:
        for seed in SEEDS:
            run_one(dataset, seed)


if __name__ == "__main__":
    main()
