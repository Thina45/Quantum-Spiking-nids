# v1.1.0 — Release Notes

This version accompanies the Scientific Reports submission *"QSNN-NIDS: A Quantum
Leaky Integrate-and-Fire Spiking Network for Intrusion Detection under Simulated
Depolarizing Noise."*

## What was added

- **Classical spiking-network ablation**: the quantum circuit inside each QLIF
  layer replaced by a parameter-matched classical current source
  (`Linear(8,3,bias=False) -> Linear(3,8,bias=False) -> sigmoid`, 96 params,
  matching the quantum ansatz's 96 params exactly; 340 total params, identical
  to QSNN-NIDS). Run on both datasets (UNSW-NB15, CIC-IDS2017) x 5 seeds
  (7, 13, 42, 99, 2024), with the same preprocessing, hyperparameters, and seed
  handling as the existing QSNN-NIDS results.
  - New code: `qsnn/models/classical_current_qlif.py`, `qsnn/models/classical_snn.py`,
    `scripts/run_classical_snn_ablation.py`.
  - New results: `results/revision/classical_snn/` (`per_seed.csv`, `summary.csv`,
    `REPORT.md`, and raw per-run output/checkpoints under `runs/`).
  - `results/revision/classical_snn/REPORT.md` additionally answers, quoting exact
    code locations: the precise per-timestep equations the quantum QLIF layer
    implements (and how they differ from a commonly assumed simplified form); which
    learning rate the optimizer actually applies to the quantum weights; and how
    each seed's data split is derived (no persisted per-seed bundle file exists for
    either the original QSNN-NIDS runs or this ablation — both rebuild the split
    deterministically from the seed).
- `.zenodo.json`: new Zenodo metadata file (title, version, creators, license).
- `README.md`: updated paper title, added a "Classical spiking-network ablation"
  section describing the experiment and its results table.

## What did NOT change

- **No existing model, training, data, or evaluation code was modified.** The
  ablation is implemented entirely as new, additive files.
- **No existing result file, checkpoint, or figure was modified or re-run.** All
  previously reported QSNN-NIDS, multi-seed validation, noise-sweep, and IBM
  hardware results are untouched.
- No raw dataset files are included in this or any prior release.

## Provenance note

A known, confirmed (not hypothesized) side effect of reusing the existing
optimizer code unmodified: the leak/threshold scalar parameters (`beta_logit`,
`thr_logit`) are routed to the optimizer's quantum-learning-rate group by name
match, even in the classical ablation where there is no quantum component. This
is documented, not corrected, per the instruction not to modify existing code to
"fix" anything — see `results/revision/classical_snn/REPORT.md`, Q2.
