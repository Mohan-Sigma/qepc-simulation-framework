# Release v1.0.0 — JxCDC submission

Reproducibility package for:

> **QEPC: Deferred-Evaluation Probabilistic Computing for Wearable AI —
> Architecture, Measured Evaluation, and Design Limits**
> Mohan Raj Manoharan
> IEEE Journal on Exploratory Solid-State Computational Devices and Circuits

This tag is the exact state of the code used to produce every table and figure
in the submitted manuscript. Later commits may change; this release will not.

---

## Before you run: two things that catch people out

### 1. Expect 20–40 minutes, not seconds

The leave-one-subject-out sweep evaluates **nine hyperparameter combinations
across fifteen folds** for each inference policy. On a typical laptop:

| Command | Approximate runtime |
|---|---|
| `preprocess_ppg_dalia_v5.py` | 2–5 minutes |
| `ppg_benchmark.py --loso` | **20–40 minutes** |
| `ppg_benchmark.py --all` | 30–50 minutes |
| `energy_tables.py` | seconds |

There is no progress bar during the LOSO sweep. It has not hung — it is
working through the grid. If you want a fast sanity check first, run

```bash
python ppg_benchmark.py --npz ppg_dalia_v5.npz --coverage
```

which finishes in a few seconds and confirms the data loaded correctly.

### 2. Windows paths: quote them, and never paste them into source

Two separate problems, both easy to hit.

**On the command line**, wrap the path in double quotes. Paths with spaces
will otherwise be split into multiple arguments:

```bat
python preprocess_ppg_dalia_v5.py --data-dir "D:\My Folder\IEEE Works\Datasets" --out ppg_dalia_v5.npz
```

**Inside Python source**, a raw Windows path will break the file before any
code runs. `D:\My\US\...` contains `\U`, which Python reads as the start of a
Unicode escape:

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes ...
truncated \UXXXXXXXX escape
```

`\N`, `\x` and `\u` fail the same way. If you must put a path in source, use a
raw string or double the backslashes:

```python
path = r"D:\My\US\Datasets"      # raw string — correct
path = "D:\\My\\US\\Datasets"    # escaped — also correct
path = "D:\My\US\Datasets"       # SyntaxError
```

The preprocessing script accepts a path argument specifically so you do not
have to edit source. Use `--data-dir`.

---

## Quick start

```bash
git clone https://github.com/<user>/qepc-simulation-framework.git
cd qepc-simulation-framework
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download **PPG-DaLiA** from the UCI Machine Learning Repository (dataset 495)
and extract it. The archive may contain a nested zip; the preprocessing script
detects and unpacks it, and locates the subject pickles recursively.

```bash
python preprocess_ppg_dalia_v5.py --data-dir "<path to extracted dataset>" --out ppg_dalia_v5.npz
python ppg_benchmark.py --npz ppg_dalia_v5.npz --loso --export-S S_loso.npz
python energy_tables.py --S S_loso.npz
```

---

## Expected output

If your run is correct, `ppg_benchmark.py --loso` reports:

| Method | MAE (BPM) | mean \|S\| |
|---|---|---|
| Naive argmax | 16.04 | 1.00 |
| Causal fixed-window (W=16) | 15.03 | 16.00 |
| QEPC deferred (H<0.30) | 14.91 | 10.87 |
| QEPC deferred (H<0.40) | 15.12 | 6.35 |
| QEPC deferred (H<0.50) | 15.63 | 4.86 |
| QEPC deferred (H<0.60) | 16.00 | 2.70 |
| Fixed-lag smoother (L=2)* | 14.43 | 18.00 |
| Non-causal smoother* | 13.21 | all |

\* not matched comparisons — these use observations later than the estimate point

`energy_tables.py` should report 1.47× at H<0.30 rising to 5.66× at H<0.60,
a 744× gap between volatile and non-volatile retention, and a compute share of
1.3% of per-query energy.

Small differences in the last decimal place across NumPy versions are normal.
Differences in the first decimal place are not — check that all 15 subjects
loaded (`64,697 windows` on the first line of output).

---

## What is not included

- **The dataset.** PPG-DaLiA is 2.7 GB and separately licensed. Download it
  from UCI; `.gitignore` blocks the raw pickles and derived `.npz` from being
  committed by accident.
- **Circuit or device simulation.** This is an operation-accounting model. It
  counts operations and prices them from published device characterisations.
  Section 15 of supporting document SD-1 states what it cannot establish.

---

## Known limitations

- Energy figures are inference-core only. Un-gated logic leakage, clock
  distribution, sensor analog front end and host interface are excluded and
  dominate total system power at this duty cycle.
- The model counts operations but does not simulate timing. No clock-frequency
  claim is made.
- The transition operator is applied densely in the code for clarity; the
  energy model prices the banded form (B·W rather than B² multiply–accumulates),
  which is what the architecture implements.
- Results are validated on chain-structured temporal models. Loopy graphs may
  mix far more slowly, and the deferred-evaluation advantage is not established
  for them.

---

## Requirements

Python 3.10 or later. Pinned versions in `requirements.txt` are those used to
produce the published numbers; looser bounds will almost certainly work.

## Licence

MIT. If you use this software, please cite the accompanying paper — see
`CITATION.cff`.
