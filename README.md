# QEPC Simulation Framework

Reproducibility package for:

> **QEPC: Deferred-Evaluation Probabilistic Computing for Wearable AI —
> Architecture, Simulation Study, and Design Limits**
> Mohan Raj Manoharan
> IEEE Journal on Exploratory Solid-State Computational Devices and Circuits

Every quantitative result in the manuscript and in supporting document SD-1
(Rev 2.0) can be regenerated with the commands below. The framework is
deterministic under a fixed seed.

---

## What this is, and what it is not

This is an **operation-accounting model**, not a circuit or device simulation.
For each algorithm and workload it counts operations of each type — state-store
reads and writes, transition applications, collapse events, inter-layer
transfers, controller invocations — and prices each using published device
characterizations. It does not simulate physics.

Transistor-level simulation at 22 nm and 40 nm requires foundry process design
kits under non-disclosure, and results produced that way could not be
reproduced by readers lacking equivalent access. What is released here is
therefore not a weaker substitute for circuit simulation but a more
reproducible one. Section 15 of SD-1 states what it cannot establish.

---

## Before you run

**Expect 20–40 minutes for the LOSO sweep, not seconds.** It evaluates nine
hyperparameter combinations across fifteen folds and prints nothing until it
finishes. It has not hung. For a fast check that the data loaded correctly,
run `python ppg_benchmark.py --npz ppg_dalia_v5.npz --coverage` first — that
returns in seconds.

**Windows users: quote your paths, and never paste one into Python source.**
On the command line, wrap the path in double quotes so spaces do not split it
into separate arguments. Inside source, a raw Windows path breaks the file
before any code runs — `D:\My\US\...` contains `\U`, which Python reads as
the start of a Unicode escape and rejects with

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes ...
```

`\N`, `\x` and `\u` fail the same way. Use a raw string (`r"D:\path"`) if you
must, but the scripts take `--data-dir` precisely so you do not have to edit
source at all.

---

## Requirements

```
python >= 3.10
numpy   >= 1.24
scipy   >= 1.11
pandas  >= 2.0     (preprocessing only)
networkx >= 3.1    (graph_generator only)
matplotlib >= 3.7  (figures only)
```

```bash
pip install numpy scipy pandas networkx matplotlib
```

---

## Data

The benchmark uses **PPG-DaLiA** (Reiss et al., *Sensors* 19(14):3079, 2019),
available from the UCI Machine Learning Repository, dataset 495. Download and
extract the archive; the preprocessing script locates the subject pickles
recursively and will unpack a nested zip if one is present.

Note that the released pickles were written under Python 2 and require
`encoding='latin1'` when unpickled under Python 3. The preprocessing script
handles this.

---

## Reproduction

### 1. Preprocess

```bash
python preprocess_ppg_dalia_v5.py --data-dir /path/to/PPG_FieldStudy \
                                  --out ppg_dalia_v5.npz
```

Produces per-window spectra for all 15 subjects: 64,697 windows, 96 spectral
bins spanning 0.6–3.5 Hz. Runtime a few minutes; output roughly 20 MB.

### 2. Accuracy — Table V, Section VI-B, Section VI-H

```bash
python ppg_benchmark.py --npz ppg_dalia_v5.npz --all --export-S S_loso.npz
```

Individual reports:

| Flag | Reproduces |
|---|---|
| `--loso` | Table V accuracy columns, leave-one-subject-out |
| `--coverage` | spectral coverage check, Section VI-B |
| `--pmf-width` | accuracy versus PMF width, SD-1 Table 4.2 |
| `--gating` | confidence-gated operation, Section VI-H |

Expect roughly 20–40 minutes for `--all` on a laptop; the leave-one-subject-out
sweep evaluates nine hyperparameter combinations across fifteen folds.

### 3. Energy — Table III, Table V energy columns, Sections VI-F and VI-G

```bash
python energy_tables.py --S S_loso.npz
```

Requires `S_loso.npz` from step 2, since the energy figures are computed from
the measured active-subgraph distributions rather than from assumed values.

---

## Expected output

Step 2 should reproduce, exactly:

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

\* not matched comparisons — these use observations later than the estimate point.

Step 3 should reproduce 1.47× at H<0.30 rising to 5.66× at H<0.60, the 744×
gap between volatile and non-volatile retention, and a compute share of 1.3%
of per-query energy.

---

## Modules

| File | Role |
|---|---|
| `preprocess_ppg_dalia_v5.py` | Raw PPG-DaLiA → per-window spectra |
| `ppg_benchmark.py` | Inference policies, LOSO harness, accuracy reports |
| `energy_model_v2.py` | Hardware energy model; selectable retention technology |
| `energy_tables.py` | Energy, retention, sensitivity and thermal reports |
| `graph_generator.py` | Probabilistic graphical model topologies |
| `qepc_inference.py` | Deferred-evaluation engine on general graphs |
| `baseline_bp.py` | Belief-propagation and simulated-annealing baselines |

`graph_generator.py`, `qepc_inference.py` and `baseline_bp.py` support the
general-graph experiments discussed in Section VII, including the traveling
salesman benchmark that produced the negative erasure result. They are not
required to reproduce the manuscript's headline tables.

---

## Method note: why the full spectrum

An earlier formulation reduced each window to its three strongest spectral
peaks and inferred which was the pulse. Measured on the full cohort, the true
heart rate fell within 5 BPM of a retained peak in only **76.6%** of windows,
so peak extraction discarded the answer outright in nearly a quarter of cases,
and no inference over the retained set could recover it.

Retaining the full spectrum resolves this. The true bin carries median
normalized magnitude 0.876 and exceeds 0.2 in 97.0% of windows, but is the
global maximum only 22.8% of the time. The task is peak selection, not peak
detection — which is what temporal probabilistic inference supplies.

Accelerometer spectral subtraction was also tried and made matters worse; the
accelerometer spectrum is frequently energetic at the true heart-rate bin, so
subtraction erodes the pulse peak in low-motion windows. `--coverage`
reproduces the check.

---

## Known limitations

- Energy figures are **inference-core only**. Un-gated logic leakage, clock
  distribution, the sensor analog front end and the host interface are
  excluded, and dominate total system power at this duty cycle.
- The model counts operations but does not simulate timing. No clock-frequency
  claim is made.
- The transition operator is applied densely here for clarity; the energy model
  prices the banded form (B·W rather than B² multiply–accumulates), which is
  what the architecture implements.
- Results are validated on chain-structured temporal models. Loopy graphs may
  mix far more slowly, and the deferred-evaluation advantage is not established
  for them.

---

## Citation

Please cite the manuscript. If you use PPG-DaLiA, cite Reiss et al. (2019) as
well.
