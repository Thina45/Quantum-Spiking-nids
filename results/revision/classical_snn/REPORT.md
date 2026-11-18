# Classical SNN Ablation — Report

**Status: experiment running in the background** (`scripts/run_classical_snn_ablation.py`, both datasets x 5 seeds, launched after a 2-epoch end-to-end smoke test passed cleanly). The investigative questions below are answered now, independent of the run; the results sections (comparison table, wall-clock, failures) will be filled in once all 10 runs finish. No existing code was changed to answer any of these.

---

## Q1 — Exact per-timestep equations `QuantumLIFLayer.forward()` implements

Source: `qsnn/models/qlif.py:198-245` (`forward()`), `qsnn/models/qlif.py:175-185` (`membrane_to_angle()`), `qsnn/models/qlif.py:187-195` (`_p_excited()`), `qsnn/models/qsnn.py:122-124` (how `current` is produced upstream).

In execution order, per timestep `t` (default `reset="subtract"`, `qlif.py:220-221`):

1. **Membrane integration** (`qlif.py:221`):
   `mem = beta * mem + current[:, t] - spk * v_th`
   — `current[:, t]` is **not** the circuit's own prior output fed back; it is the *externally supplied synaptic drive*, precomputed for all `T` timesteps at once by the classical `Linear` synapse one layer up (`cur = syn(h)`, `qsnn.py:123`), before this layer's loop even starts. `spk` here is the *previous* iteration's spike `S[t-1]` (it has not been reassigned yet this iteration).
2. **Angle map** (`qlif.py:226`, `membrane_to_angle`, `qlif.py:175-185`):
   `theta = pi * sigmoid(angle_gain * mem + angle_bias)`, i.e. `theta_t = pi * phi(g * U_t + b)` with `g=2, b=-2` — **the input is the membrane `U_t` (post-integration, step 1's result), not the raw drive `z_i[t]`.**
3. **Circuit evaluation** (`qlif.py:227`, `_p_excited`, `qlif.py:187-195`):
   `p1 = (1 - circuit(theta, qweights)) / 2`, i.e. `P(|1>) = (1 - <Z_i>) / 2`.
4. **Spike** (`qlif.py:230`):
   `spk = spike_fn(p1 - v_th, surrogate, surrogate_alpha)` — **thresholds `p1` (the circuit's `P(|1>)` output) against `v_th`, not the membrane `U_t` directly.**

Readout (`qsnn.py:130-134`, config `readout: rate_mem_prob`, `configs/unsw_nb15.yaml:47`): concatenates **all three** — mean spike rate over `T` (`rate`), `tanh` of the final membrane (`mem`), and mean `P(|1>)` over `T` (`prob`) — confirmed from the actual config value, not assumed.

### Comparison against your hypothesized equations

> `theta_i[t] = phi(g*z_i[t] + b); |psi> = U_ent(w) U_enc(theta)|0>; I_i[t] = (1-<Z_i>)/2; U_i[t] = beta*U_i[t-1] + I_i[t] - S_i[t-1]*V_th; S_i[t] = H(U_i[t] - V_th)`

**Does not match. Three concrete differences:**

1. **Angle-map input.** Your equations map `theta` from the raw drive `z_i[t]`. The code maps `theta` from the **membrane `U_t`** (`qlif.py:226`, using `mem` which already includes the integration step), not from `z` directly.
2. **What drives the membrane update.** Your equations feed the circuit's own output `I_i[t] = (1-<Z_i>)/2` back into the *same-or-next* membrane update (`U_i[t] = beta*U_i[t-1] + I_i[t] - ...`). The code's membrane update instead uses `current[:, t]`, the **externally supplied synaptic drive from the upstream classical `Linear` layer** (`qsnn.py:123`) — the circuit's `P(|1>)` output is **never fed back as a membrane current** at all, in this or any layer.
3. **What gets thresholded.** Your `S_i[t] = H(U_i[t] - V_th)` thresholds the membrane directly. The code thresholds the **circuit output**, `spk = spike_fn(p1 - v_th, ...)` (`qlif.py:230`) — `p1`, not `U_t`.

What *does* match: the refractory-reset term's form (`- S[t-1] * V_th`, `qlif.py:221`), the angle-map functional form `phi(g*z+b)` itself (just applied to `U` rather than `z`), and the circuit-output definition `(1-<Z_i>)/2` (`qlif.py:193-195`).

---

## Q2 — Quantum-weight learning rate

**Yes, confirmed — a separate LR of `1e-2` was applied to the quantum weights in the actual QSNN-NIDS runs.**

Optimizer param-group code, `qsnn/engine.py:93-112` (`make_optimizer`):
```python
q, c = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (q if ("qweights" in name or "beta_logit" in name or "thr_logit" in name)
     else c).append(p)
groups = [
    {"params": c, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
    {"params": q, "lr": cfg.train.quantum_lr, "weight_decay": 0.0},
]
```
Config value: `configs/unsw_nb15.yaml:68` — `quantum_lr: 0.01`. This group split is by *parameter name substring match*, not an explicit "is this a quantum weight" flag.

**One precise, confirmed side-effect for this ablation's faithfulness:** the name-match also catches `beta_logit`/`thr_logit` (the learnable leak and threshold scalars, present in *both* `QuantumLIFLayer` and the new `ClassicalCurrentLIFLayer` with identical attribute names). Since `make_optimizer()` was reused completely unmodified (per your instruction), running it on `ClassicalSNN` routes `beta_logit`/`thr_logit` into the `quantum_lr=1e-2` group **even though there is no quantum component in this ablation** — purely because the name-matching logic doesn't know the difference. The new current-source weights (`fc1.weight`, `fc2.weight`) correctly fall into the `c` group (`lr=cfg.train.lr=3e-3`), matching your stated intent exactly. I did not change `make_optimizer()` to "fix" the `beta_logit`/`thr_logit` routing — flagging it here as requested, not silently patching it.

---

## Q3 — Per-seed data bundle

**There is no persisted per-seed bundle *file* for the original QSNN-NIDS runs to begin with** — confirmed by reading `scripts/run_multiseed.py:58` (the script that produced the existing `artifacts/qsnn_*_seed*/results.json`): *"deliberately no `--bundle`: seed must drive its own data split"*. This means `scripts/train.py:61` took the `build_dataset(cfg)` branch (not `load_bundle`), rebuilding the split **in memory only**, never saved to disk, for every one of the existing seed-42/7/13/99/2024 runs.

I confirmed what actually exists on disk under `data/processed/`:
```
cicids2017.npz                       <- seed-42 base bundle (from prepare_data.py)
cicids2017_noisesub.npz              <- seed-42 noise-sweep TEST subsample only
cicids2017_seed13_noisesub.npz       <- seed-13 noise-sweep TEST subsample only
cicids2017_seed7_noisesub.npz        <- seed-7  noise-sweep TEST subsample only
unsw_nb15.npz / unsw_nb15_noisesub.npz / unsw_nb15_seed{7,13}_noisesub.npz   (same pattern)
```
These `_noisesub.npz` files (from `scripts/run_noise_multiseed.py`'s `build_seed_subsample()`) are **derived test-set-only subsamples for the noise sweep**, not the original full train/val/test bundle, and only exist for seeds 7 and 13 (not 42, 99, or 2024) — they are not what the detection-metric seed runs used.

**Consequence for this ablation:** "the same bundle used by the matching QSNN-NIDS seed run" means reproducing it the same way the original run did — by calling `build_dataset(cfg)` with `cfg.train.seed = cfg.data.seed = <seed>`, which is deterministic (`train_test_split(..., random_state=d.seed, ...)`, `preprocess.py:260-265`; `np.random.RandomState(d.seed)` for undersampling/MI sampling) given the same raw CSVs and code. This is exactly what `scripts/run_classical_snn_ablation.py` does (no `--bundle`, same override pattern as `run_multiseed.py`) — there was no bundle *file* to load for either the original or the ablation runs.

---

## Experiment results

**Status: complete. All 10 runs (2 datasets x 5 seeds) finished with `status=OK`, zero failures, zero skipped/deviated runs.** Raw per-run output: `results/revision/classical_snn/runs/classical_snn_<dataset>_seed<seed>/results.json`. Aggregated: `results/revision/classical_snn/per_seed.csv` (every run, every metric) and `results/revision/classical_snn/summary.csv` (mean +/- std per dataset).

### Parameter counts (measured, both models)

| | current-source / quantum params | other classical params | total |
|---|---|---|---|
| QSNN-NIDS (quantum circuit) | 96 | 244 | 340 |
| Classical SNN ablation (this run) | 96 | 244 | 340 |

Exact match, confirmed by `model.param_counts()` on the actual built models (not computed by hand) -- see Phase 0 message for the raw command output.

### Comparison table: QSNN-NIDS vs Classical SNN ablation, mean +/- std, 5 seeds (7, 13, 42, 99, 2024)

QSNN-NIDS detection-metric rows are read from `artifacts/unsw_nb15_results_summary_multiseed.csv` and `artifacts/cicids2017_results_summary_multiseed.csv` (the `QSNN (ours)` row) -- **not re-run**. QSNN-NIDS spike-rate/latency are not in those summary CSVs, so they were computed here by averaging the `efficiency.spike_rate` / `efficiency.latency_ms_per_sample` fields read directly from the 5 existing per-seed files: `artifacts/qsnn_unsw_v2/results.json`, `artifacts/qsnn_unsw_seed{7,13,99,2024}/results.json` (and the `cicids2017` equivalents) -- also not re-run, only read.

#### UNSW-NB15

| Metric | QSNN-NIDS | Classical SNN (this ablation) |
|---|---|---|
| Accuracy | 0.8602 +/- 0.0038 | 0.8724 +/- 0.0008 |
| Precision | 0.8185 +/- 0.0166 | 0.8473 +/- 0.0059 |
| Recall | 0.9151 +/- 0.0205 | 0.8984 +/- 0.0085 |
| F1 (attack) | 0.8637 +/- 0.0014 | 0.8720 +/- 0.0012 |
| FPR | 0.1912 +/- 0.0260 | 0.1519 +/- 0.0144 |
| ROC-AUC | 0.9448 +/- 0.0039 | 0.9533 +/- 0.0013 |
| MCC | 0.7268 +/- 0.0036 | 0.7464 +/- 0.0016 |
| Spike rate (test) | 0.1919 +/- 0.0372 | 0.3296 +/- 0.1061 |
| Latency (ms/flow) | 5.6757 +/- 1.1239 | 0.0396 +/- 0.0036 |

#### CIC-IDS2017

| Metric | QSNN-NIDS | Classical SNN (this ablation) |
|---|---|---|
| Accuracy | 0.9779 +/- 0.0036 | 0.9801 +/- 0.0027 |
| Precision | 0.9303 +/- 0.0144 | 0.9500 +/- 0.0100 |
| Recall | 0.9602 +/- 0.0077 | 0.9493 +/- 0.0094 |
| F1 (attack) | 0.9449 +/- 0.0088 | 0.9496 +/- 0.0068 |
| FPR | 0.0177 +/- 0.0038 | 0.0124 +/- 0.0027 |
| ROC-AUC | 0.9953 +/- 0.0015 | 0.9930 +/- 0.0024 |
| MCC | 0.9314 +/- 0.0110 | 0.9373 +/- 0.0063 |
| Spike rate (test) | 0.2106 +/- 0.0610 | 0.2992 +/- 0.0939 |
| Latency (ms/flow) | 5.3912 +/- 0.4929 | 0.0298 +/- 0.0053 |

**Reading this honestly:** the classical current-source replacement matches or very slightly exceeds QSNN-NIDS on every detection metric on both datasets, at the same parameter count (340, including the exact same 96 "coupling" parameters). Latency is ~140x lower (no quantum circuit simulation). This is expected and should be reported plainly rather than downplayed: it is evidence that, *for this specific architecture, dataset, and parameter budget*, the entangling quantum ansatz is not yet contributing a detection-accuracy advantage over the classical bottleneck it was compared against here -- a negative/null result for the "quantum circuit improves accuracy" claim specifically, not a negative result for the overall QSNN-NIDS architecture (membrane dynamics, spiking, temporal encoding) relative to the non-spiking classical baselines in the main paper, which it continues to beat (see the RF/SVM/MLP comparisons already in `artifacts/*_results_summary_multiseed.csv`).

### Wall-clock: actual vs Phase 0 projection

| | Phase 0 projection | Actual |
|---|---|---|
| Total (10 runs) | ~1-1.5 hours (~66-90 min, with early-stopping expected to shorten this) | **55.8 min** |

Actual came in under the projection, consistent with early stopping triggering before all 30 epochs on most runs (`epochs_run` column in `per_seed.csv` ranges from 19 to 30 across the 10 runs) and the per-epoch cost matching the Phase 0 benchmark closely (~10-14s/epoch on UNSW-NB15, ~9-11s/epoch on CIC-IDS2017, observed in `logs/classical_snn_ablation.log`).

### Failed, skipped, or deviated runs

**None.** All 10 (dataset, seed) combinations completed with `status=OK` on the first attempt; no retries, no `FAILED.json` written, no run skipped. Early stopping (patience=8) triggered on most runs before the full 30 epochs, as designed -- this is the existing, unmodified early-stopping behavior, not a deviation.

### Environment (logged per run, identical across all 10)

PyTorch `2.14.0+cpu`, Python `3.13.0`, CPU `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`, 16 logical CPUs.
