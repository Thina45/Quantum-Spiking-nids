#!/usr/bin/env python3
"""
ONE FILE, start to finish, for Kaggle (or any machine).

What it does, in order:
  1. installs requirements (skipped if already satisfied)
  2. runs the offline smoke test (fails fast if the environment is broken)
  3. finds the UNSW-NB15 CSVs under /kaggle/input (or --root) and preprocesses them
  4. trains the QSNN + all baselines (RF, SVM, MLP, QSVC, QCNN) on the identical split
  5. flattens the results into ONE CSV you can open in Excel/Sheets directly

Usage on Kaggle (after attaching the mrwellsdavid/unsw-nb15 dataset):
    !cd /kaggle/working/qsnn_nids && python scripts/kaggle_run_all.py

Usage locally (already-downloaded CSVs under data/raw/unsw_nb15/):
    python scripts/kaggle_run_all.py --config configs/unsw_nb15.yaml

The single result file lands at:
    artifacts/<run_name>/results_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def sh(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def find_unsw_root(explicit: str | None) -> str:
    """Locate the folder holding UNSW_NB15_training-set.csv / testing-set.csv."""
    if explicit:
        return explicit
    candidates = [
        "data/raw/unsw_nb15",
        "/kaggle/input/unsw-nb15",
        "/kaggle/input/unsw-nb15/UNSW-NB15",
    ]
    candidates += glob.glob("/kaggle/input/*unsw*", recursive=False)
    candidates += glob.glob("/kaggle/input/**/UNSW_NB15_training-set.csv", recursive=True)
    for c in candidates:
        root = c if os.path.isdir(c) else os.path.dirname(c)
        hit = glob.glob(os.path.join(root, "**", "UNSW_NB15_training-set.csv"), recursive=True)
        if hit:
            return root
    raise SystemExit(
        "could not find UNSW_NB15_training-set.csv anywhere under data/raw/unsw_nb15 "
        "or /kaggle/input -- pass --root explicitly."
    )


def flatten_results(results_json_path: str, out_csv_path: str) -> None:
    """One row per model, detection metrics + (for the QSNN) the resource/efficiency
    metrics merged in as extra columns -- so the whole comparison is one flat table."""
    import csv

    with open(results_json_path) as fh:
        payload = json.load(fh)

    rows = payload["comparison"]
    efficiency = payload.get("qsnn", {}).get("efficiency", {})

    all_keys: list[str] = []
    flat_rows = []
    for r in rows:
        row = {"model": r["model"], "family": r["family"], **r.get("metrics", {})}
        if r["model"] == "QSNN (ours)":
            row.update(efficiency)
        flat_rows.append(row)
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        for row in flat_rows:
            w.writerow(row)
    print(f"[io]    single-file summary -> {out_csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unsw_nb15.yaml")
    ap.add_argument("--root", default=None, help="folder with the UNSW-NB15 CSVs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--baselines", nargs="*", default=["rf", "svm", "mlp", "qsvc", "qcnn"])
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-smoke-test", action="store_true")
    a = ap.parse_args()

    py = sys.executable

    if not a.skip_install:
        sh([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    if not a.skip_smoke_test:
        sh([py, "scripts/smoke_test.py"])

    root = find_unsw_root(a.root)
    print(f"[data]  using UNSW-NB15 root: {root}")

    sh([py, "scripts/prepare_data.py", "--config", a.config, "--root", root])

    bundle = os.path.join(
        "data", "processed",
        os.path.splitext(os.path.basename(a.config))[0] + ".npz",
    )
    train_cmd = [
        py, "scripts/train.py", "--config", a.config,
        "--bundle", bundle, "--epochs", str(a.epochs),
    ]
    if a.baselines:
        train_cmd += ["--baselines", *a.baselines]
    sh(train_cmd)

    from qsnn.config import Config
    cfg = Config.from_yaml(a.config)
    out_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    flatten_results(
        os.path.join(out_dir, "results.json"),
        os.path.join(out_dir, "results_summary.csv"),
    )

    print("\nDONE. One-file results table:")
    print(f"  {os.path.join(out_dir, 'results_summary.csv')}")
    print("Open it in Excel/Sheets -- one row per model (QSNN + every baseline),")
    print("detection metrics in every row, quantum-resource columns filled in")
    print("only on the QSNN row (that's the paper's differentiator).")


if __name__ == "__main__":
    main()
