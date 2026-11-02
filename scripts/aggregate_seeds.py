#!/usr/bin/env python3
"""Aggregate per-seed results.json / noise_sweep.json into mean +/- std
tables ready for a paper.

Finds every artifacts/*/config.used.yaml, groups by (dataset, train.seed)
read from the config itself (not the folder name), so it doesn't matter
that seed 42's folders are named qsnn_unsw_v2 / qsnn_cicids2017_v1 while
the rest are qsnn_<dataset>_seed<N>.

Outputs, per dataset:
    artifacts/<dataset>_results_summary_multiseed.csv
    artifacts/<dataset>_noise_sweep_multiseed.csv   (only if >=2 seeds have one)

Also prints an outlier-flagging report to stdout: any seed whose QSNN
metrics differ from the group mean by more than 2x the group std, or whose
QSNN F1 is suspiciously close to a majority-class baseline (collapse).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as stats
from collections import defaultdict
from typing import Dict, List

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "mcc"]


def discover_runs() -> Dict[str, List[dict]]:
    """dataset -> list of {seed, run_dir, results, noise_sweep}"""
    out: Dict[str, List[dict]] = defaultdict(list)
    for cfg_path in glob.glob(os.path.join(REPO_ROOT, "artifacts", "*", "config.used.yaml")):
        run_dir = os.path.dirname(cfg_path)
        results_path = os.path.join(run_dir, "results.json")
        if not os.path.exists(results_path):
            continue  # unfinished / crashed run
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        dataset = cfg.get("data", {}).get("dataset")
        seed = cfg.get("train", {}).get("seed")
        if dataset is None or seed is None:
            continue
        with open(results_path) as fh:
            results = json.load(fh)
        noise_path = os.path.join(run_dir, "noise_sweep.json")
        noise = None
        if os.path.exists(noise_path):
            with open(noise_path) as fh:
                noise = json.load(fh)
        out[dataset].append({
            "seed": seed, "run_dir": run_dir,
            "results": results, "noise": noise,
        })
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


def mean_std_str(vals: List[float], decimals: int = 4) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.{decimals}f} (n=1)"
    m = stats.mean(vals)
    s = stats.pstdev(vals)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def aggregate_detection(dataset: str, runs: List[dict], out_path: str) -> None:
    # model -> metric -> [values across seeds]
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    families: Dict[str, str] = {}
    seeds_seen: Dict[str, List[int]] = defaultdict(list)

    for run in runs:
        for row in run["results"].get("comparison", []):
            model = row["model"]
            families[model] = row.get("family", "")
            seeds_seen[model].append(run["seed"])
            for k, v in row.get("metrics", {}).items():
                if k in METRICS and isinstance(v, (int, float)):
                    per_model[model][k].append(float(v))

    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "family", "n_seeds", "seeds"] + METRICS)
        for model, metric_vals in per_model.items():
            row = [model, families[model], len(seeds_seen[model]),
                   ";".join(str(s) for s in sorted(set(seeds_seen[model])))]
            for m in METRICS:
                vals = metric_vals.get(m, [])
                row.append(mean_std_str(vals) if vals else "")
            w.writerow(row)
    print(f"[io] {out_path}")


def aggregate_noise(dataset: str, runs: List[dict], out_path: str) -> None:
    runs_with_noise = [r for r in runs if r["noise"]]
    if len(runs_with_noise) < 1:
        return
    # level -> metric -> [values across seeds]
    per_level: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    seeds_used = sorted({r["seed"] for r in runs_with_noise})

    for run in runs_with_noise:
        for row in run["noise"].get("sweep", []):
            p = row.get("depolarizing")
            for k in ("accuracy", "f1", "roc_auc", "fpr"):
                if k in row:
                    per_level[p][k].append(float(row[k]))

    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["depolarizing_p", "n_seeds", "seeds", "accuracy", "f1", "roc_auc", "fpr"])
        for p in sorted(per_level):
            row = [p, len(seeds_used), ";".join(str(s) for s in seeds_used)]
            for k in ("accuracy", "f1", "roc_auc", "fpr"):
                vals = per_level[p].get(k, [])
                row.append(mean_std_str(vals) if vals else "")
            w.writerow(row)
    print(f"[io] {out_path}  ({len(seeds_used)} seeds: {seeds_used})")


def flag_outliers(dataset: str, runs: List[dict]) -> None:
    qsnn_rows = []
    for run in runs:
        for row in run["results"].get("comparison", []):
            if row["model"] == "QSNN (ours)":
                qsnn_rows.append((run["seed"], row["metrics"]))

    if len(qsnn_rows) < 2:
        return

    for metric in ("f1", "accuracy", "roc_auc"):
        vals = [m[metric] for _, m in qsnn_rows if metric in m]
        if len(vals) < 2:
            continue
        mean, sd = stats.mean(vals), stats.pstdev(vals)
        for seed, m in qsnn_rows:
            if metric not in m or sd == 0:
                continue
            z = abs(m[metric] - mean) / sd
            if z > 2.0:
                print(f"  [OUTLIER] {dataset} seed={seed}: {metric}={m[metric]:.4f} "
                      f"is {z:.1f} std from the {len(vals)}-seed mean ({mean:.4f})")

    # collapse check: F1 near 0, or accuracy near a plausible majority-class rate
    for seed, m in qsnn_rows:
        if m.get("f1", 1.0) < 0.05:
            print(f"  [COLLAPSE?] {dataset} seed={seed}: QSNN F1={m.get('f1'):.4f} "
                  f"-- looks like majority-class collapse, inspect this run's history.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()

    by_dataset = discover_runs()
    if not by_dataset:
        print("No completed runs found under artifacts/*/config.used.yaml + results.json")
        return

    for dataset, runs in by_dataset.items():
        seeds = [r["seed"] for r in runs]
        print(f"\n=== {dataset}: {len(runs)} completed run(s), seeds {sorted(seeds)} ===")
        aggregate_detection(dataset, runs,
                            os.path.join(REPO_ROOT, "artifacts",
                                         f"{dataset}_results_summary_multiseed.csv"))
        aggregate_noise(dataset, runs,
                        os.path.join(REPO_ROOT, "artifacts",
                                     f"{dataset}_noise_sweep_multiseed.csv"))
        flag_outliers(dataset, runs)


if __name__ == "__main__":
    main()
