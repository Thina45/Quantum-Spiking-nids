# Session Memory — QSNN-NIDS Project

This file records what was done in this Claude Code session so future sessions (human or AI) have continuity. `PROJECT_LOG.md` (dated 2026-10-15) is stale and predates everything below — do not treat it as authoritative.

---

## 1. Starting point

Prior to this session (earlier work, not detailed here): fixed a UNSW-NB15 test-CSV loader bug, fixed a CIC-IDS2017 latin1-encoding bug, added resumable/checkpointed training to `qsnn/engine.py` and `scripts/train.py`, ran full QSNN + 5-baseline training on both UNSW-NB15 (`artifacts/qsnn_unsw_v2`, seed 42) and CIC-IDS2017 (`artifacts/qsnn_cicids2017_v1`, seed 42), ran 6-level noise sweeps on both at seed 42, and published two Claude Artifacts (a validation-run report and a draft-review comparing `QML_IDS.pdf` against real results).

## 2. Qniverse SDK integration (unresolved / paused)

Goal: three-way quantum backend (local simulator / IBM hardware / Qniverse-C-DAC HPC), explicitly preserving the IBM Quantum connection.

- `pip install qniverse` and variants (`qniverse-sdk`, `qsdk`, `Qniverse`, etc.) all fail — package not on public PyPI despite docs claiming otherwise. User's own Qniverse dashboard had no SDK download either.
- Pivoted to direct REST calls using browser-captured credentials (JWT + WordPress session cookie, both provided by the user from their own logged-in session — **treat these as sensitive, never commit them**).
- Confirmed: the Qniverse API (`qniverse.in/QACC-Backend/...`) uses **WordPress session-cookie auth**, not Bearer/API-key headers — all header-based auth attempts returned the HTML login page.
- With the cookie, got real JSON back, but `provider=superstaq` requires a separate Superstaq API key (not configured) and the native QACC backend submission needs a `Cluster_Name` value that was never successfully determined (`QACC_Cluster`, `QACC-Cluster`, and case variants all rejected).
- **Left open:** asked the user to capture the real "Submit" button API call via DevTools → "Copy as cURL" to finish reverse-engineering the native backend contract. This was never followed up — conversation pivoted to the multi-seed task. **Resume here if Qniverse integration is picked back up.**
- Known API shape (from official docs, confirmed structurally correct even though install failed): `from Qniverse.backend import Backend`; `token` must be a global set before `Backend()` instantiation; config order must be `.processor()` → `.hpc()` → `.simulator()`; circuits via `QniverseCircuit`/`QuantumRegister`/`ClassicalRegister`/`add_gate`; jobs via `run(qc, backend, shots, token=token, job_name=...)`.

## 3. IBM Quantum hardware

- `scripts/run_ibm_hardware.py` exists and is ready to use (targets `ibm_fez`, 3 samples, 256 shots, using `artifacts/qsnn_unsw_v2/final.pt`).
- Retried twice in this session — both times hit: `UserWarning: This instance has met its usage limit. Workloads will not run until time is made available.`
- **Status: still blocked by account quota**, no ETA. Worth retrying again closer to submission, not worth polling repeatedly.

## 4. GitHub repository setup — DONE

- User created `https://github.com/Thina45/Quantum-Spiking-nids` (public, Apache 2.0) via GitHub's web UI.
- Fixed `.gitignore` (two bugs caught before any bad commit — see `git log` / earlier transcript for exact diffs). Final working version excludes `data/raw/*`, `data/processed/*` (keeping `.gitkeep`s), `logs/`, `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store` — everything in `artifacts/` is tracked (all tiny, KB-scale).
- Hit and fixed a `git pull --allow-unrelated-histories` issue (GitHub auto-created a LICENSE-only commit at repo creation with no common ancestor to the local initial commit).
- **Push succeeded.** Repo is live and up to date as of the git-setup portion of this session (does **not** yet include the multi-seed results produced afterward — see open items below).

## 5. Multi-seed validation sweep — DONE, see full report

Ran the full multi-seed validation requested for the Scientific Reports submission:

- **Training:** QSNN + RF/SVM/MLP/QCNN (QSVC excluded — too slow to repeat, ~2-2.65h/run) at 4 new seeds (7, 13, 99, 2024) × 2 datasets = 8 combos, via `scripts/run_multiseed.py`. All 8 completed successfully. Seed-42 results/checkpoints were never touched.
  - Important technical note: `--seed` reseeds **both** model init and `data.seed`, so every seed gets a genuinely different train/val/test split (not just different weight init) — this was flagged to the user before launching, per their explicit request.
  - The job was paused/resumed multiple times across the session (laptop sleep); it is fully resumable via checkpoint + `results.json`-exists skip logic. No progress was lost except mid-epoch state on interruption (re-ran that epoch from scratch, cheap).
- **Noise sweep:** extended to seeds 7 and 13 (in addition to existing seed 42) on both datasets via `scripts/run_noise_multiseed.py`. Hit and fixed one real bug: the script was saving `meta` as a Python `str(dict)` repr instead of JSON, breaking `load_bundle()`'s `json.loads` call — fixed by using `json.dumps` instead (see script, function `build_seed_subsample`). All 4 combos completed after the fix.
- **Aggregation:** `scripts/aggregate_seeds.py` produced:
  - `artifacts/unsw_nb15_results_summary_multiseed.csv`, `artifacts/cicids2017_results_summary_multiseed.csv` — mean±std detection metrics, 5 seeds.
  - `artifacts/unsw_nb15_noise_sweep_multiseed.csv`, `artifacts/cicids2017_noise_sweep_multiseed.csv` — mean±std noise metrics, 3 seeds.
  - Zero outliers/collapses flagged on the core detection task.

### Key findings (full detail in `artifacts/MULTISEED_VALIDATION_REPORT.md`)

1. **QSNN's detection performance is stable across seeds** — F1 std = 0.0014 (UNSW) / 0.0088 (CIC-IDS2017) across 5 seeds. Confirms the original seed-42 numbers already in the paper draft are representative, not a lucky outlier.
2. **Noise robustness is stable up to p≈0.02**, but **at p=0.05/0.1 there is a real bimodal collapse**, not just variance: QSNN drops to majority-class prediction (F1≈0) in 4 of 6 (dataset, seed) combinations at high noise, while ROC-AUC often stays well above chance — i.e. the decision *threshold* breaks before the underlying score separability does. Seed 13/UNSW is the one seed that stays robust at both high-noise levels. **This should NOT be reported as a plain mean±std for p≥0.05** — it would hide the bimodality. The full report has per-seed numbers and suggested manuscript language for this caveat.

**Full written report:** `artifacts/MULTISEED_VALIDATION_REPORT.md` — contains complete tables, the outlier/collapse analysis, and a ready-to-adapt manuscript paragraph.

## 6. Current open items / what's next

- [ ] **Push the new multi-seed artifacts and report to GitHub** — the push done earlier (section 4) predates all of section 5's results. Not yet committed/pushed.
- [ ] **PDF rendering of the report** — no PDF toolchain (`pandoc`, `weasyprint`, `reportlab`) is installed; report currently exists only as Markdown. User can ask to install `reportlab` and convert if a PDF is needed.
- [ ] **Zenodo deposit** for processed `.npz` bundles + checkpoints + results, for a permanent DOI in the paper's Data/Code Availability section. Needs user-side account setup first (like GitHub repo creation did). Not started.
- [ ] **IBM hardware retry** — still blocked by quota (section 3), retry again later, no urgency.
- [ ] **Qniverse integration** — paused mid-reverse-engineering (section 2), needs the DevTools cURL capture to continue, if the user wants to pick it back up.
- Explicitly **not** this session's responsibility: Table S1 wording fix, reproducibility checklist, submission mechanics/cover letter — user is handling those separately with help elsewhere. User's framing: "you keep your running side... we will do this" — i.e. keep the technical/running side moving, don't take over paper-writing/submission logistics.

## 7. Scripts added this session (safe, wrapper-only — core `qsnn/engine.py` and `qsnn/data/datasets.py` were never modified, per explicit user instruction)

- `scripts/run_multiseed.py` — orchestrates training across new seeds/datasets, idempotent/resumable.
- `scripts/run_noise_multiseed.py` — orchestrates noise sweep across new seeds, builds per-seed stratified subsamples.
- `scripts/aggregate_seeds.py` — discovers all completed runs, aggregates into mean±std CSVs, flags outliers/collapses.
