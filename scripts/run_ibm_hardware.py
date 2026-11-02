#!/usr/bin/env python3
"""One real-hardware validation pass on IBM Quantum.

Deliberately tiny: hardware queue time/cost scales with circuit evaluations,
and the checklist only asks to confirm the model runs on real hardware, not
to re-measure full-dataset accuracy there. Evaluates a handful of test points
on an actual IBM backend and reports per-sample predictions + wall time.
"""
from __future__ import annotations

import argparse
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
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--ibm-backend", default="ibm_fez")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--out", default="artifacts/ibm_hardware_run.json")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config)
    set_seed(cfg.train.seed)
    cfg.noise.enabled = True
    cfg.noise.backend = "ibm_runtime"
    cfg.noise.ibm_backend = a.ibm_backend
    cfg.noise.shots = a.shots

    bundle = load_bundle(a.bundle)
    cfg.model.n_classes = bundle.n_classes
    device = resolve_device(cfg.train.device)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    model = build_model(cfg, bundle.n_features)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(cfg.train.seed)
    idx = rng.choice(len(bundle.X_test), size=min(a.n_samples, len(bundle.X_test)), replace=False)
    X = torch.as_tensor(bundle.X_test[idx], dtype=torch.float32).to(device)
    y_true = bundle.y_test[idx]

    print(banner(f"IBM HARDWARE RUN  |  backend={a.ibm_backend}  |  n={len(idx)}  |  shots={a.shots}"))
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(X)
    secs = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    y_pred = probs.argmax(1)
    acc = float((y_pred == y_true).mean())

    print(f"  wall time        : {secs:.1f}s for {len(idx)} samples ({secs / len(idx):.2f}s/sample)")
    print(f"  accuracy (n={len(idx)}) : {acc:.4f}")
    for i in range(len(idx)):
        print(f"    sample {i:>2}  true={int(y_true[i])}  pred={int(y_pred[i])}  p={probs[i].round(3).tolist()}")

    save_json({
        "backend": a.ibm_backend, "shots": a.shots, "n_samples": len(idx),
        "wall_seconds": secs, "accuracy": acc,
        "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
        "probs": probs.tolist(),
    }, a.out)


if __name__ == "__main__":
    main()
