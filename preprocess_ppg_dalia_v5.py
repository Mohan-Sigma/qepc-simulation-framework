#!/usr/bin/env python3
r"""
preprocess_ppg_dalia_v5.py
==========================
Final feature extraction for the QEPC PPG-DaLiA benchmark.

WHAT CHANGED FROM v4, AND WHY
-----------------------------
v4 applied accelerometer spectral subtraction before peak picking. Measured
on S1+S2 this made things worse -- candidate coverage fell from 85.0% (raw)
to 80.4% (cleaned). The cause was a bug: each spectrum was normalised to unit
maximum independently, so the accelerometer spectrum peaked at 1.0 whether
the subject was sprinting or sitting still. Subtracting it removed a similar
amount of energy regardless of actual motion, gouging the pulse peak in quiet
windows. The accelerometer spectrum is energetic at the true-HR bin in 40.7%
of windows, so this was actively destructive.

Diagnosis on the v4 spectra showed the real situation: the pulse IS present
in the raw PPG spectrum (median magnitude 0.877 at the true-HR bin; above 0.2
in 97.4% of windows) but is the global maximum only 21% of the time. The task
is peak SELECTION, not peak detection or denoising.

The formulation that works is therefore to skip candidate extraction and use
the whole spectrum as the HMM emission, letting temporal continuity do the
selection. On S1+S2 this gives MAE 8.88 BPM (median 2.13) against 11.55 for
naive argmax, with DeepPPG's published figure at 7.65 BPM.

v5 accordingly drops the failed subtraction and the candidate arrays, keeping
only the spectra the working model consumes. This also brings the 15-subject
output down to roughly 12-22 MB, within upload limits.

OUTPUT
------
    hr_true      (T,)      ground-truth HR from chest ECG, BPM
    subject      (T,)      subject label
    window_idx   (T,)
    activity     (T,)
    spec_ppg     (T, 96)   normalised PPG magnitude spectrum   [float16]
    spec_acc     (T, 96)   normalised accelerometer spectrum   [float16]
    freqs        (96,)     bin centre frequencies, Hz

Pass --ppg-only to omit spec_acc if the file is still too large to upload.

REQUIREMENTS
------------
    pip install numpy scipy

USAGE (Windows -- keep the quotes)
----------------------------------
    python preprocess_ppg_dalia_v5.py --data-dir "D:\path\to\Datasets" --out ppg_dalia_v5.npz

    # if the resulting file exceeds the upload limit
    python preprocess_ppg_dalia_v5.py --data-dir "D:\path\to\Datasets" --ppg-only --out ppg_dalia_v5.npz
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import zipfile
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


FS_BVP, FS_ACC, FS_ACTIVITY = 64, 32, 4
WIN_SEC, SHIFT_SEC = 8, 2
BAND_LO_HZ, BAND_HI_HZ = 0.6, 3.5
N_BINS, NFFT = 96, 4096

PKL_PATTERN = re.compile(r"^(S\d{1,2})\.pkl$", re.IGNORECASE)


# ─── File discovery ──────────────────────────────────────────────────────────

def find_subject_pickles(root: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in filenames:
            m = PKL_PATTERN.match(fn)
            if m:
                subj = m.group(1).upper()
                path = os.path.join(dirpath, fn)
                if subj not in found or len(path) < len(found[subj]):
                    found[subj] = path
    return found


def auto_extract(root: str, max_rounds: int = 4) -> None:
    for _ in range(max_rounds):
        pending = []
        for dirpath, _dn, fns in os.walk(root):
            for f in fns:
                if f.lower().endswith(".zip"):
                    z = os.path.join(dirpath, f)
                    tgt = os.path.splitext(z)[0]
                    if not (os.path.isdir(tgt) and os.listdir(tgt)):
                        pending.append(z)
        if not pending:
            return
        for z in pending:
            tgt = os.path.splitext(z)[0]
            print(f"  extracting {os.path.basename(z)} ...", end="", flush=True)
            try:
                os.makedirs(tgt, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(tgt)
                print(" done")
            except Exception as e:
                print(f" FAILED ({type(e).__name__})")


def sort_subjects(s: List[str]) -> List[str]:
    return sorted(s, key=lambda x: int(x[1:]))


# ─── Signal processing ───────────────────────────────────────────────────────

def bandpass(x: np.ndarray, fs: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    lo, hi = BAND_LO_HZ / nyq, min(BAND_HI_HZ / nyq, 0.99)
    if len(x) < 3 * (order + 1):
        return x
    b, a = butter(order, [lo, hi], btype="band")
    try:
        return filtfilt(b, a, x)
    except ValueError:
        return x


def band_spectrum(x: np.ndarray, fs: int, edges: np.ndarray) -> np.ndarray:
    """Magnitude spectrum on the shared grid, normalised to unit maximum."""
    if len(x) < 8:
        return np.zeros(N_BINS)
    x = np.asarray(x, float)
    x = x - x.mean()
    if not np.any(np.abs(x) > 0):
        return np.zeros(N_BINS)

    spec = np.abs(np.fft.rfft(x, n=NFFT))
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / fs)

    out = np.zeros(N_BINS)
    idx = np.digitize(freqs, edges) - 1
    valid = (idx >= 0) & (idx < N_BINS)
    np.maximum.at(out, idx[valid], spec[valid])

    mx = out.max()
    return out / mx if mx > 0 else out


# ─── Extraction ──────────────────────────────────────────────────────────────

def load_subject(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def process_subject(subject: str, path: str, edges: np.ndarray,
                    ppg_only: bool) -> Dict[str, np.ndarray]:
    data = load_subject(path)

    bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten().astype(float)
    acc = np.asarray(data["signal"]["wrist"]["ACC"]).astype(float)
    hr  = np.asarray(data["label"]).flatten().astype(float)
    activity = np.asarray(data.get("activity", [])).flatten().astype(float)

    acc_mag = np.linalg.norm(acc, axis=1) if acc.ndim == 2 else np.abs(acc).flatten()
    bvp_f = bandpass(bvp, FS_BVP)
    acc_f = bandpass(acc_mag, FS_ACC)

    SP, SA, HR, ACT, WIN = [], [], [], [], []

    for i in range(len(hr)):
        t0 = i * SHIFT_SEC
        b0, b1 = t0 * FS_BVP, (t0 + WIN_SEC) * FS_BVP
        a0, a1 = t0 * FS_ACC, (t0 + WIN_SEC) * FS_ACC
        if b1 > len(bvp_f) or a1 > len(acc_f):
            break

        SP.append(band_spectrum(bvp_f[b0:b1], FS_BVP, edges))
        if not ppg_only:
            SA.append(band_spectrum(acc_f[a0:a1], FS_ACC, edges))

        act_idx = int((t0 + WIN_SEC / 2) * FS_ACTIVITY)
        ACT.append(float(activity[act_idx]) if act_idx < len(activity) else np.nan)
        HR.append(hr[i])
        WIN.append(i)

    out = {
        "hr_true":    np.array(HR, dtype=np.float32),
        "activity":   np.array(ACT, dtype=np.float32),
        "window_idx": np.array(WIN, dtype=np.int32),
        "spec_ppg":   np.array(SP, dtype=np.float16),
        "subject":    np.array([subject] * len(HR)),
    }
    if not ppg_only:
        out["spec_acc"] = np.array(SA, dtype=np.float16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="PPG-DaLiA spectral features (v5)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="ppg_dalia_v5.npz")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--ppg-only", action="store_true",
                    help="Omit the accelerometer spectrum to reduce file size")
    ap.add_argument("--auto-extract", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    if args.auto_extract:
        auto_extract(root)

    available = find_subject_pickles(root)
    if not available:
        print("ERROR: no S<number>.pkl found. Try --auto-extract.", file=sys.stderr)
        return 1

    wanted = (sort_subjects([s.strip().upper()
                             for s in args.subjects.split(",") if s.strip()])
              if args.subjects.strip() else sort_subjects(list(available)))
    wanted = [s for s in wanted if s in available]

    edges = np.linspace(BAND_LO_HZ, BAND_HI_HZ, N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    parts = []
    for subj in wanted:
        print(f"  [{subj}] processing ...", end="", flush=True)
        try:
            p = process_subject(subj, available[subj], edges, args.ppg_only)
            parts.append(p)
            print(f" {len(p['hr_true']):,} windows")
        except Exception as e:
            print(f" FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    if not parts:
        print("Nothing processed.", file=sys.stderr)
        return 1

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    merged["freqs"] = centres.astype(np.float32)

    out_path = os.path.abspath(args.out)
    np.savez_compressed(out_path, **merged)
    size_mb = os.path.getsize(out_path) / 1e6

    hr = merged["hr_true"]
    P = merged["spec_ppg"].astype(float)
    naive = centres[np.argmax(P, axis=1)] * 60
    mae_naive = float(np.abs(naive - hr).mean())

    print(f"\nWrote {out_path}")
    print(f"  windows  : {len(hr):,}")
    print(f"  subjects : {len(np.unique(merged['subject']))}")
    print(f"  size     : {size_mb:.1f} MB")
    print(f"  naive argmax MAE : {mae_naive:.2f} BPM  (reference floor)")
    if size_mb > 28:
        print("\n  NOTE: file may exceed the upload limit. Re-run with --ppg-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
