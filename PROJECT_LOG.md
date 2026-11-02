# Project log: Quantum Spiking NIDS (QSNN-NIDS)

Folder analysed: `V:\D\Quantum programs\Quantum Apriori Algorithm\Quantum spiking nids`. The folder was read in
full (30 files, including the Kaggle notebook output) and **nothing was run or changed**. Date of analysis: 2026-10-15.

Project: a quantum spiking neural network for network intrusion detection (team of three students; guide
Parthiban Krishnamoorthy). It is not a git repository, and `artifacts/` and `data/` are empty.

---

## 1. What it is
- **Model:** 8 qubits form 2 QLIF layers (quantum leaky integrate-and-fire); one qubit is one neuron.
  Each timestep is integrate, rotate (RY), entangle, measure. Spikes are trained with surrogate gradients.
  Built with PennyLane and PyTorch.
- **Pipeline:** spike encoders (phase, rate, latency, delta, repeat); quantile scaling; mutual-information filter;
  PCA to 8 dimensions; baselines (Random Forest, SVM, MLP, quantum-kernel SVM, QCNN); noise-sweep and ablation scripts.
- **Claimed contribution:** the first QSNN for intrusion detection, with a circuit whose depth does not grow with
  the number of timesteps.

## 2. Datasets (links provided)
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html (needs the registration form on that page)
- Neither has been downloaded in this workspace. Nothing has been trained or evaluated here on either.

## 3. The only real result (`qml-notebook_results.ipynb`, Kaggle CPU, run Sep 7)
UNSW-NB15 binary classification, 50,000 sampled rows, 7,284 test rows, 8 qubits, 2 QLIF layers, phase coding, T=12.
Training stopped early at epoch 23 (about 232 s per epoch).

| Model | Accuracy | F1 | FPR | ROC-AUC |
|---|---|---|---|---|
| QSNN | 0.8585 | 0.8408 | 0.1698 | 0.9397 |
| Random Forest | 0.9334 | 0.9216 | 0.0717 | 0.9856 |
| RBF-SVM | 0.9279 | 0.9156 | 0.0799 | 0.9776 |
| MLP | 0.9099 | 0.8962 | 0.1072 | 0.9750 |

Footprint: depth 15 and 32 two-qubit gates per timestep, 24 circuit evaluations per flow, spike rate 0.284,
3.5-3.8 ms per flow in simulation. The smoke test passed 11 of 11 checks (plumbing only, on synthetic data).

Not run at all: noise sweep, classical-LIF ablation, encoding and qubit ablations, CIC-IDS2017, the quantum
baselines (QSVC, QCNN) on real data, and any real quantum hardware.

## 4. Findings, most serious first
1. The QSNN loses to every classical baseline on every metric. The decisive ablation (quantum versus classical
   LIF neuron, already coded in `qlif.py`) was never run, so there is no evidence the qubit adds anything.
2. The test set is a random split of the training file only. `_find_files` in `datasets.py` stops at the first
   matching pattern, so `UNSW_NB15_testing-set.csv` is never loaded. Results are not comparable with the literature.
3. The reported numbers use a simulator-only readout: the exact mean P(|1>) and deterministic threshold spikes. The
   measurement-faithful stochastic mode is off by default and was not evaluated.
4. The hardware claim is untested. Depth is constant in T, but each flow needs 24 sequential measure-and-prepare
   round trips that cannot be batched within a flow, so real-device latency would be seconds to a minute per flow
   (an order-of-magnitude estimate, not a measurement).
5. The sequence mode for CIC-IDS2017 (`make_flow_windows`) is not wired into `build_dataset`, and the random split
   would break time order anyway.
6. One seed and no confidence intervals; validation F1 swings between 0.80 and 0.85 from epoch to epoch.
7. PCA to 8 dimensions (91.5% of variance) caps accuracy for all models, so comparison with 99% literature figures
   is not like for like.
8. "No published QSNN for intrusion detection" is an unverified absolute claim; it needs a documented search or
   hedged wording.
9. "Sparsity saving 71.6%" is misleading for a quantum circuit: each timestep's circuit costs the same whether or
   not a spike occurs.
10. Housekeeping: no git repository; the trained checkpoint exists only in the Kaggle session.

## 5. What is solid
- Leakage-aware preprocessing: split first, fit every transform on the training part only.
- The monotone angle-map fix and the BatchNorm readout fix.
- FPR and efficiency metrics reported alongside accuracy.
- The test set is used once, with early stopping on validation F1.

## 6. Suggested next steps (need your go; this is the team's code)
1. Fix the file loader so both official UNSW-NB15 files are used, and rerun.
2. Run the quantum-versus-classical ablation with at least 5 seeds and report intervals.
3. Add trivial baselines (predicting everything as an attack gives F1 of about 0.59 on this test set).
4. Run the noise sweep and a shot-based, stochastic-spike evaluation to approximate hardware.
5. Download CIC-IDS2017 for the cross-dataset test.
6. Optionally, a small real-hardware check of one timestep circuit.
