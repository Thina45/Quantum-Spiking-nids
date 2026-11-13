#!/usr/bin/env python3
"""Single real-hardware validation pass for the paper: noise-free inference
(the real device's native noise IS the experiment -- no depolarizing sweep
layered on top), on a small stratified subset of the UNSW-NB15 test set,
using the already-trained seed-42 weights. Compares hardware predictions
against the same subset run through the noise-free local simulator.

Does not retrain, does not touch any existing seed-42 or multi-seed
artifacts -- everything is written to a new artifacts/qsnn_unsw_v2/
ibm_hardware_run/ directory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from qsnn.config import Config
from qsnn.data.preprocess import load_bundle
from qsnn.engine import resolve_device, set_seed
from qsnn.metrics import detection_metrics
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, save_json


def stratified_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
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
    return idx[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unsw_nb15.yaml")
    ap.add_argument("--checkpoint", default="artifacts/qsnn_unsw_v2/final.pt")
    ap.add_argument("--bundle", default="data/processed/unsw_nb15.npz")
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--ibm-backend", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/qsnn_unsw_v2/ibm_hardware_run")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    cfg = Config.from_yaml(a.config)
    set_seed(cfg.train.seed)

    bundle = load_bundle(a.bundle)
    cfg.model.n_classes = bundle.n_classes
    device = resolve_device(cfg.train.device)

    idx = stratified_indices(bundle.y_test, a.n_samples, a.seed)
    X = torch.as_tensor(bundle.X_test[idx], dtype=torch.float32).to(device)
    y_true = bundle.y_test[idx]
    classes, counts = np.unique(y_true, return_counts=True)

    print(banner(f"IBM HARDWARE RUN  |  backend={a.ibm_backend}  |  n={len(idx)}  |  shots={a.shots}"))
    print(f"  class balance in subset: {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"  class balance full test: "
          f"{dict(zip(*np.unique(bundle.y_test, return_counts=True)))}")

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # ---- noise-free LOCAL SIMULATOR pass (same weights, same flows) -------
    cfg_sim = Config.from_yaml(a.config)
    cfg_sim.model.n_classes = bundle.n_classes
    cfg_sim.noise.enabled = False
    model_sim = build_model(cfg_sim, bundle.n_features)
    model_sim.load_state_dict(state, strict=True)
    model_sim.to(device)
    model_sim.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits_sim = model_sim(X)
    sim_secs = time.perf_counter() - t0
    probs_sim = torch.softmax(logits_sim, dim=-1).cpu().numpy()
    y_pred_sim = probs_sim.argmax(1)
    print(f"  [simulator] done in {sim_secs:.2f}s")

    # ---- REAL HARDWARE pass -------------------------------------------------
    cfg_hw = Config.from_yaml(a.config)
    cfg_hw.model.n_classes = bundle.n_classes
    cfg_hw.noise.enabled = True
    cfg_hw.noise.backend = "ibm_runtime"
    cfg_hw.noise.ibm_backend = a.ibm_backend
    cfg_hw.noise.shots = a.shots
    model_hw = build_model(cfg_hw, bundle.n_features)
    model_hw.load_state_dict(state, strict=True)
    model_hw.to(device)
    model_hw.eval()

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            logits_hw = model_hw(X)
    except Exception as exc:
        err_path = os.path.join(a.out_dir, "FAILED.json")
        save_json({
            "status": "FAILED", "error": str(exc), "error_type": type(exc).__name__,
            "backend": a.ibm_backend, "shots": a.shots, "n_samples": len(idx),
            "timestamp": datetime.datetime.now().isoformat(),
        }, err_path)
        print(banner("IBM HARDWARE RUN FAILED"))
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  details saved -> {err_path}")
        raise SystemExit(1)
    hw_secs = time.perf_counter() - t0
    probs_hw = torch.softmax(logits_hw, dim=-1).cpu().numpy()
    y_pred_hw = probs_hw.argmax(1)
    print(f"  [hardware]  done in {hw_secs:.2f}s ({hw_secs / len(idx):.2f}s/sample wall time, includes queue)")

    # ---- metrics --------------------------------------------------------
    m_hw = detection_metrics(y_true, y_pred_hw, probs_hw, n_classes=bundle.n_classes)
    m_sim = detection_metrics(y_true, y_pred_sim, probs_sim, n_classes=bundle.n_classes)
    agree = int((y_pred_hw == y_pred_sim).sum())
    agreement_frac = agree / len(idx)

    print(banner("RESULTS"))
    print(f"  hardware : acc={m_hw['accuracy']:.4f}  f1={m_hw['f1']:.4f}")
    print(f"  simulator: acc={m_sim['accuracy']:.4f}  f1={m_sim['f1']:.4f}")
    print(f"  agreement (hw vs sim predictions): {agree}/{len(idx)} ({agreement_frac:.4f})")

    out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "backend": a.ibm_backend,
        "shots": a.shots,
        "n_samples": len(idx),
        "sampling_seed": a.seed,
        "flow_indices": idx.tolist(),
        "class_balance_subset": {int(k): int(v) for k, v in zip(classes, counts)},
        "class_balance_full_test": {
            int(k): int(v) for k, v in zip(*np.unique(bundle.y_test, return_counts=True))
        },
        "y_true": y_true.tolist(),
        "hardware": {
            "wall_seconds": hw_secs,
            "y_pred": y_pred_hw.tolist(),
            "probs": probs_hw.tolist(),
            "metrics": m_hw,
        },
        "simulator": {
            "wall_seconds": sim_secs,
            "y_pred": y_pred_sim.tolist(),
            "probs": probs_sim.tolist(),
            "metrics": m_sim,
        },
        "agreement": {
            "n_matching": agree,
            "n_total": len(idx),
            "fraction": agreement_frac,
        },
    }
    out_path = os.path.join(a.out_dir, "ibm_hardware_run.json")
    save_json(out, out_path)
    print(f"\n[io] saved -> {out_path}")


if __name__ == "__main__":
    main()
