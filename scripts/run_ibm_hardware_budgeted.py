#!/usr/bin/env python3
"""Real-hardware validation with a hard, enforced device-time budget.

Processes ONE flow at a time through the existing (working) single-circuit
hardware path -- this sidesteps the batching bug documented in
artifacts/IBM_HARDWARE_VALIDATION_REPORT.md (the ibm_runtime device does not
batch across flows), by never asking it to. Before every real job
submission, checks actual account usage via QiskitRuntimeService().usage()
and refuses to submit if doing so could plausibly exceed --budget-seconds.

--dry-run: exercises the entire pipeline (flow selection, local simulator
comparison, budget-tracking logic, output saving) with FAKE hardware
responses. No network calls that could consume quota. Must be run and
checked before a real (non---dry-run) invocation.
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

CIRCUIT_EVALS_PER_FLOW = 24  # T(12) * n_qlif_layers(2), see qsnn/models/qsnn.py resource_report()
OBSERVED_SECONDS_PER_JOB = 11.0  # from job dbqtvhejjaos73ahv0qg and others this session
SAFETY_MARGIN = 1.3  # treat each job as if it costs 30% more than observed, for the stop decision


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


def get_usage_seconds(service) -> float:
    """Real account usage, with retry-with-backoff (this is what broke last
    time -- a single DNS blip must not crash the whole run)."""
    last_exc = None
    for attempt in range(5):
        try:
            return float(service.usage()["usage_consumed_seconds"])
        except Exception as exc:  # network hiccup -- retry, don't crash
            last_exc = exc
            wait = 2 ** attempt
            print(f"    [usage-check retry {attempt + 1}/5] {type(exc).__name__}: {exc}  (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"could not read account usage after 5 retries: {last_exc}")


def run_one_flow_hardware(model_hw, x_row: torch.Tensor, dry_run: bool, n_classes: int = 2):
    """Runs exactly one flow through the model (batch size 1), on hardware
    (or faked, in --dry-run). Returns (probs[1,C], wall_seconds)."""
    if dry_run:
        time.sleep(0.05)  # simulate some latency, no network
        fake = torch.softmax(torch.randn(1, n_classes), dim=-1)
        return fake.numpy(), 0.05
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model_hw(x_row.unsqueeze(0))
    secs = time.perf_counter() - t0
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unsw_nb15.yaml")
    ap.add_argument("--checkpoint", default="artifacts/qsnn_unsw_v2/final.pt")
    ap.add_argument("--bundle", default="data/processed/unsw_nb15.npz")
    ap.add_argument("--n-flows", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--ibm-backend", required=True)
    ap.add_argument("--budget-seconds", type=float, required=True,
                     help="hard cap on ADDITIONAL real device-seconds this run may consume")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/qsnn_unsw_v2/ibm_hardware_run")
    ap.add_argument("--dry-run", action="store_true",
                     help="exercise the full pipeline with fake hardware responses, no network calls")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    cfg = Config.from_yaml(a.config)
    set_seed(cfg.train.seed)
    bundle = load_bundle(a.bundle)
    cfg.model.n_classes = bundle.n_classes
    device = resolve_device(cfg.train.device)

    idx_all = stratified_indices(bundle.y_test, a.n_flows, a.seed)
    X_all = torch.as_tensor(bundle.X_test[idx_all], dtype=torch.float32).to(device)
    y_true_all = bundle.y_test[idx_all]

    print(banner(
        f"IBM HARDWARE RUN (budgeted)  |  backend={a.ibm_backend}  |  "
        f"n_flows_requested={len(idx_all)}  |  shots={a.shots}  |  "
        f"budget={a.budget_seconds}s  |  dry_run={a.dry_run}"
    ))

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # ---- local simulator pass on the FULL requested set (free, always real) ----
    cfg_sim = Config.from_yaml(a.config)
    cfg_sim.model.n_classes = bundle.n_classes
    cfg_sim.noise.enabled = False
    model_sim = build_model(cfg_sim, bundle.n_features)
    model_sim.load_state_dict(state, strict=True)
    model_sim.to(device)
    model_sim.eval()
    with torch.no_grad():
        logits_sim = model_sim(X_all)
    probs_sim_all = torch.softmax(logits_sim, dim=-1).cpu().numpy()
    y_pred_sim_all = probs_sim_all.argmax(1)
    print(f"  [simulator] ran on all {len(idx_all)} requested flows (free, local)")

    # ---- hardware device (real network call to fetch backend metadata; no shots yet) ----
    service = None
    model_hw = None
    if not a.dry_run:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
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

    usage_start = 0.0 if a.dry_run else get_usage_seconds(service)
    print(f"  account usage at start of this run: {usage_start:.1f}s consumed")

    completed = []
    stopped_early = False
    stop_reason = None

    for i, (orig_idx, x_row, y_t) in enumerate(zip(idx_all, X_all, y_true_all)):
        projected_cost = CIRCUIT_EVALS_PER_FLOW * OBSERVED_SECONDS_PER_JOB * SAFETY_MARGIN
        if a.dry_run:
            used_so_far = sum(f["wall_seconds_proxy"] for f in completed)
        else:
            used_so_far = get_usage_seconds(service) - usage_start
        if used_so_far + projected_cost > a.budget_seconds:
            stopped_early = True
            stop_reason = (
                f"stopping before flow {i} (index {int(orig_idx)}): "
                f"used {used_so_far:.1f}s so far, next flow projected to cost up to "
                f"{projected_cost:.1f}s (24 jobs x {OBSERVED_SECONDS_PER_JOB}s x {SAFETY_MARGIN} margin), "
                f"budget is {a.budget_seconds}s"
            )
            print(f"  [BUDGET STOP] {stop_reason}")
            break

        print(f"  [flow {i}] index={int(orig_idx)}  submitting {CIRCUIT_EVALS_PER_FLOW} real circuit jobs"
              f"{' (DRY RUN - no network)' if a.dry_run else ''} ...")
        try:
            probs_hw, secs = run_one_flow_hardware(model_hw, x_row, a.dry_run, bundle.n_classes)
        except Exception as exc:
            err_path = os.path.join(a.out_dir, f"FAILED_flow{i}.json")
            save_json({
                "status": "FAILED", "flow_index_in_run": i, "orig_test_index": int(orig_idx),
                "error": str(exc), "error_type": type(exc).__name__,
                "timestamp": datetime.datetime.now().isoformat(),
            }, err_path)
            print(banner(f"FLOW {i} FAILED"))
            print(f"  {type(exc).__name__}: {exc}")
            print(f"  details saved -> {err_path}")
            print(f"  {len(completed)} flow(s) completed successfully before this failure -- "
                  f"those results are still saved.")
            break

        y_pred_hw = int(probs_hw.argmax(1)[0])
        completed.append({
            "flow_index_in_run": i,
            "orig_test_index": int(orig_idx),
            "y_true": int(y_t),
            "y_pred_hardware": y_pred_hw,
            "probs_hardware": probs_hw[0].tolist(),
            "y_pred_simulator": int(y_pred_sim_all[i]),
            "probs_simulator": probs_sim_all[i].tolist(),
            "wall_seconds_proxy": secs if a.dry_run else projected_cost,
        })
        print(f"    done  true={int(y_t)}  hw_pred={y_pred_hw}  sim_pred={int(y_pred_sim_all[i])}  "
              f"match={'yes' if y_pred_hw == int(y_pred_sim_all[i]) else 'NO'}")

    # ---- aggregate whatever completed ----
    n_done = len(completed)
    if n_done > 0:
        y_true_done = np.array([c["y_true"] for c in completed])
        y_pred_hw_done = np.array([c["y_pred_hardware"] for c in completed])
        y_pred_sim_done = np.array([c["y_pred_simulator"] for c in completed])
        agree = int((y_pred_hw_done == y_pred_sim_done).sum())
        m_hw = detection_metrics(y_true_done, y_pred_hw_done, n_classes=bundle.n_classes) if n_done > 1 else None
        m_sim = detection_metrics(y_true_done, y_pred_sim_done, n_classes=bundle.n_classes) if n_done > 1 else None
    else:
        agree, m_hw, m_sim = 0, None, None

    out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dry_run": a.dry_run,
        "backend": a.ibm_backend,
        "shots": a.shots,
        "budget_seconds": a.budget_seconds,
        "n_flows_requested": len(idx_all),
        "n_flows_completed": n_done,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "flows": completed,
        "agreement": {"n_matching": agree, "n_total": n_done,
                      "fraction": (agree / n_done) if n_done else None},
        "metrics_hardware": m_hw,
        "metrics_simulator": m_sim,
    }
    out_path = os.path.join(a.out_dir, ("DRYRUN_" if a.dry_run else "") + "ibm_hardware_run_budgeted.json")
    save_json(out, out_path)

    print(banner("SUMMARY"))
    print(f"  requested: {len(idx_all)}  completed: {n_done}  stopped_early: {stopped_early}")
    if n_done:
        print(f"  agreement (hw vs sim): {agree}/{n_done}")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
