# IBM Quantum Hardware Validation Report

**Date:** 2026-11-12
**Project:** QSNN-NIDS
**Purpose:** Obtain genuine real-QPU inference data for the paper, using the already-trained seed-42 UNSW-NB15 weights, under a strict, user-enforced device-time budget.

**Outcome: partial success.** One flow was fully and correctly evaluated on real IBM Quantum hardware, within budget, with its prediction matching the local noise-free simulator. A second attempt earlier in this session failed before this; a real, previously-undocumented limitation in the pipeline's hardware-batching code was also discovered and is reported honestly below rather than worked around under time pressure.

---

## 1. Context: two attempts

### Attempt 1 (failed, see Section 4 for the root cause it exposed)
- Config: 30 flows, 96 shots, `ibm_fez`, intended 5-minute budget.
- Result: local simulator pass succeeded; hardware pass crashed on a transient DNS resolution failure (`Failed to resolve 'quantum.cloud.ibm.com'`) after only 1 of the 24 circuit-evaluation jobs needed for a single flow had completed.
- This also surfaced a deeper issue: the completed job returned a result shaped for a *single* circuit, not a batch of 30 — see Section 4.
- Quota consumed: 11s (~0.17 RU) under the old 10-RU/600s quota period.

### Attempt 2 (this report's main result)
- The account's IBM Quantum quota was upgraded mid-session: from 600s/10 RU (resetting monthly) to **10,800s (3 hours) / 180 RU, resetting yearly** (period: 2026-11-12 → 2027-11-12).
- Despite the much larger available quota, the user explicitly capped this run to **10 minutes (600s) of real device time**, reserving the rest for actual PhD work rather than spending it on this validation pass.
- A new script, `scripts/run_ibm_hardware_budgeted.py`, was written and **dry-run tested end-to-end before any real submission** (flow selection, simulator comparison, budget-tracking logic, output saving — all verified with zero network calls, catching and fixing one real bug: a `None`-model crash in the dry-run fake-response path).
- The script processes one flow at a time (sidestepping the Attempt-1 batching limitation entirely — see Section 4), checking real account usage via `QiskitRuntimeService().usage()` before every flow and refusing to start a new one if doing so could plausibly exceed the 600s cap (using a conservative 24-jobs × 11s/job × 1.3 safety-margin projection per flow).

## 2. Attempt 2 result

| | |
|---|---|
| Backend | `ibm_fez` |
| Shots/circuit | 96 |
| Budget | 600s (hard cap) |
| Flows requested | 5 |
| **Flows completed** | **1** |
| Stopped early | Yes — budget-stop triggered correctly before flow 2 |
| **Real device time consumed** | **264s (4.4 min)** — within the 600s cap |
| Account usage after | 264s / 4.11 RU out of 10,800s / 180 RU (10,536s remaining) |

The loop stopped before attempting a second flow because, after flow 1's real 264s cost, the conservative projection for a second flow (≈343s) would have put the run at risk of exceeding 600s — exactly the intended safety behavior, even though in practice there was still ~336s of real budget left unused. This is deliberately conservative: it undershoots the budget rather than risk overshooting it.

### The one completed flow

| | True class | Hardware prediction | Hardware confidence | Simulator prediction | Simulator confidence | Match |
|---|---|---|---|---|---|---|
| Test-set index 7409 | 0 (benign) | **0** | 0.9991 | **0** | 0.9561 | ✅ Yes |

Real IBM Quantum hardware (`ibm_fez`) correctly classified this flow as benign, in agreement with the noise-free local simulator, with even higher confidence than the simulator produced (0.999 vs 0.956) — consistent with the hardware's native noise not being large enough at this shot count to flip this particular, fairly clear-cut example.

## 3. What this does and doesn't demonstrate

- **Demonstrates:** the QSNN-NIDS model runs correctly end-to-end on real IBM Quantum hardware for genuine single-flow inference, with the hardware-derived prediction matching the simulator on real device noise, within a strict and fully-respected time budget.
- **Does not demonstrate:** hardware accuracy, F1, or any aggregate robustness statistic — a sample size of 1 cannot support those claims, and no additional flows were attempted once the budget-stop logic (correctly) intervened.

## 4. Known limitation: batched (multi-flow) hardware submission does not work

Discovered during Attempt 1 and unresolved by design (left for future work, not patched under time pressure): the `ibm_runtime` branch of `qsnn/noise.py` (lines 64-72) constructs its device via `qml.device("qiskit.remote", wires=n_wires, backend=backend, shots=...)`. This device does **not** support PennyLane's automatic parameter-broadcasting across a batch of flows the way the local `default.qubit` simulator does (verified locally: a batched input of shape `[30, 8]` through the same circuit correctly returns `[30, 8]` on `default.qubit`, but the one real hardware job in Attempt 1 returned only a single `(8,)`-shaped result for an intended 30-flow batch).

**Workaround used here:** `scripts/run_ibm_hardware_budgeted.py` sidesteps this entirely by processing exactly one flow per model forward pass (batch size 1), so the batching bug never applies — this is why only one flow fits per full real-time budget cycle (24 real circuit jobs per flow, ~264s at the observed per-job rate), rather than why the batch bug caused incorrect results.

**Not fixed this session:** a proper fix would mean modifying `qsnn/noise.py` to explicitly construct one Primitive Unified Bloc (PUB) per flow within each of the 24 jobs (keeping job count at 24 regardless of flow count, each job just carrying N PUBs) — this is a core-file change, and per standing instruction it was not made without explicit approval. The user chose to document this as a known limitation rather than approve the fix this session.

## 5. Recommended language for the manuscript

> "We validated QSNN-NIDS inference on real IBM Quantum hardware (`ibm_fez`) using the trained seed-42 model weights. Under a strict device-time budget, one UNSW-NB15 test flow was evaluated end-to-end on real hardware; its predicted class matched the noise-free local simulator's prediction (both correctly classified the flow as benign), with the hardware prediction showing comparable confidence to the simulator (0.999 vs 0.956) despite real device noise. We identify a current limitation in our PennyLane-based hardware integration: unlike local simulation, batched (multi-flow) circuit submission is not yet supported for the IBM Runtime backend, limiting large-scale real-hardware evaluation under present tooling to sequential single-flow submission. Extending the integration to submit each flow as an explicit Primitive Unified Bloc within a single batched job — which would not increase the number of hardware jobs required — is identified as future work, together with a full real-hardware accuracy and robustness evaluation once that capability is in place."

## 6. Quota status after this session's attempts

| | Seconds | RU |
|---|---|---|
| Total allocation (new period) | 10,800s | 180 |
| Consumed (Attempt 1, partial) | 11s | 0.17 |
| Consumed (Attempt 2) | 264s | 4.11 |
| **Total consumed** | **275s** | **4.28** |
| **Remaining** | **10,525s (~2.9 hours)** | **~175.7** |
| Period resets | 2027-11-12 | |

Substantial real device-time budget remains for future, properly-scoped real-hardware work (e.g. after a `qsnn/noise.py` batching fix, or for further single-flow validations).

## 7. Artifacts produced

| File | Contents |
|---|---|
| `scripts/run_ibm_hardware_real.py` | Attempt 1's script (30-flow attempt, failed on DNS + exposed the batching bug) |
| `scripts/run_ibm_hardware_budgeted.py` | Attempt 2's script — per-flow processing, real-time usage-budget enforcement, dry-run tested, retry-with-backoff on the usage-check call |
| `artifacts/qsnn_unsw_v2/ibm_hardware_run/FAILED.json` | Attempt 1's exact DNS failure details |
| `artifacts/qsnn_unsw_v2/ibm_hardware_run/ibm_hardware_run_budgeted.json` | Attempt 2's full result: flow index, true label, hardware + simulator predictions and probabilities, budget accounting |
| `artifacts/qsnn_unsw_v2/ibm_hardware_run/DRYRUN_ibm_hardware_run_budgeted.json` | Dry-run verification output (fake data, confirms pipeline correctness before the real submission) |

## 8. Open items

- [ ] Decide whether to approve the `qsnn/noise.py` batching fix for a larger-scale real-hardware evaluation in the future (quota is no longer the blocker — ~2.9 hours remain).
- [ ] If pursued, re-run with a larger `--budget-seconds` and more flows using the now-fixed batching, for a genuine hardware accuracy/F1/agreement-rate comparison.
- [ ] Until then, report only the single-flow validation result (Section 2) in the paper, with the limitation language in Section 5 — do not characterize this as a full real-hardware accuracy evaluation.
