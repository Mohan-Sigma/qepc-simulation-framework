"""
ppg_benchmark.py
================
PPG-DaLiA heart-rate tracking benchmark for the QEPC deferred-evaluation
architecture. Reproduces Table V and Figs. 1-3 of the manuscript.

METHOD
------
Each 8 s window is represented by its magnitude spectrum over 0.6-3.5 Hz,
discretized into B = 96 bins of 0.030 Hz (~1.8 BPM). Inference is a
first-order hidden Markov chain:

    hidden state  z_t = heart-rate bin
    observation   o_t = normalized PPG spectrum

    emission    p(o_t | z_t = i)      ~  [S_t(f_i)]^alpha
    transition  p(z_t = i | z_{t-1}=j) ~  exp[-(f_i - f_j)^2 / 2 sigma^2]

Note that candidate peak extraction is NOT used. An earlier formulation kept
only the top-3 spectral peaks per window; the true heart rate fell within
5 BPM of a retained peak in just 76.6% of windows, so extraction discarded
the answer outright in nearly a quarter of cases. Retaining the full spectrum
resolves this: the true bin carries median normalized magnitude 0.877 and
exceeds 0.2 in 97.4% of windows, but is the global maximum only 21% of the
time. The task is peak selection, not peak detection.

POLICIES COMPARED
-----------------
  naive       argmax of the spectrum, no inference          (1 activation)
  causal      fixed-window forward filter, width W          (W activations)
  deferred    QEPC: recruit history until posterior entropy
              falls below a threshold                       (data-dependent)
  fixedlag    causal filter plus L windows of lookahead     (reference only)
  smoother    full forward-backward, non-causal             (reference only)

Only `naive`, `causal` and `deferred` are matched comparisons: they use no
observation later than the estimate point. The other two are upper bounds.

VALIDATION
----------
The model has no trained weights. The two hyperparameters (sigma, alpha) are
chosen by leave-one-subject-out cross-validation, matching the protocol of the
reference implementation [Reiss 2019].

USAGE
-----
    python ppg_benchmark.py --npz ppg_dalia_v5.npz --loso
    python ppg_benchmark.py --npz ppg_dalia_v5.npz --gating
    python ppg_benchmark.py --npz ppg_dalia_v5.npz --pmf-width
    python ppg_benchmark.py --npz ppg_dalia_v5.npz --export-S S_loso.npz
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Defaults ────────────────────────────────────────────────────────────────
W_BASELINE = 16          # causal fixed-window width
MAX_DEPTH = 24           # retention depth / deferred traversal cap
READOUT_HALFWIDTH = 3    # bins either side of the MAP bin for posterior mean

SIGMA_GRID = (0.03, 0.04, 0.06)     # transition drift, Hz
ALPHA_GRID = (1.5, 2.0, 3.0)        # emission sharpening

THRESHOLDS = (0.30, 0.40, 0.50, 0.60)


# ─── Data ────────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    hr: np.ndarray          # (N,)   ground-truth heart rate, BPM
    spec: np.ndarray        # (N, B) normalized PPG magnitude spectrum
    freqs: np.ndarray       # (B,)   bin center frequencies, Hz
    subject: np.ndarray     # (N,)   subject label
    activity: np.ndarray    # (N,)

    @property
    def subjects(self) -> List[str]:
        return sorted(np.unique(self.subject), key=lambda s: int(str(s)[1:]))

    def mask(self, subj) -> np.ndarray:
        return self.subject == subj


def load(npz_path: str) -> Dataset:
    d = np.load(npz_path, allow_pickle=True)
    return Dataset(
        hr=d['hr_true'].astype(float),
        spec=d['spec_ppg'].astype(float),
        freqs=d['freqs'].astype(float),
        subject=d['subject'],
        activity=d['activity'].astype(float) if 'activity' in d else None,
    )


# ─── Model ───────────────────────────────────────────────────────────────────

def emission(spec: np.ndarray, alpha: float) -> np.ndarray:
    """Row-normalized emission matrix, (T, B)."""
    E = np.maximum(spec, 1e-12) ** alpha
    return E / E.sum(axis=1, keepdims=True)


def transition(freqs: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian drift kernel over heart-rate bins, (B, B), row-normalized.

    In hardware this is applied as a banded operator: the kernel is negligible
    beyond +/-3 sigma, so only B*W multiply-accumulates are required rather
    than B^2. The dense form is used here for clarity; the energy model prices
    the banded form.
    """
    step = freqs[1] - freqs[0]
    idx = np.arange(len(freqs))
    D = (idx[:, None] - idx[None, :]) * step
    A = np.exp(-0.5 * (D / sigma) ** 2)
    return A / A.sum(axis=1, keepdims=True)


def band_width(freqs: np.ndarray, sigma: float, n_sigma: int = 3) -> int:
    """Effective transition bandwidth W used by the energy model."""
    step = freqs[1] - freqs[0]
    return 2 * int(np.ceil(n_sigma * sigma / step)) + 1


def _readout(belief: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Posterior mean over a window around the MAP bin, converted to BPM."""
    B = belief.shape[1]
    k = np.argmax(belief, axis=1)
    out = np.empty(len(k))
    for i, ki in enumerate(k):
        lo, hi = max(0, ki - READOUT_HALFWIDTH), min(B, ki + READOUT_HALFWIDTH + 1)
        w = belief[i, lo:hi]
        out[i] = (w @ freqs[lo:hi]) / max(w.sum(), 1e-12) * 60.0
    return out


def _norm_entropy(belief: np.ndarray) -> np.ndarray:
    B = belief.shape[1]
    p = np.maximum(belief, 1e-12)
    return -(p * np.log(p)).sum(axis=1) / np.log(B)


# ─── Policies ────────────────────────────────────────────────────────────────

def policy_naive(spec: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    est = freqs[np.argmax(spec, axis=1)] * 60.0
    return est, np.ones(len(est), dtype=int)


def policy_causal(E: np.ndarray, A: np.ndarray, freqs: np.ndarray,
                  W: int = W_BASELINE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-window causal forward filter. At each t the filter runs over
    [t-W+1, t]; no observation later than t is used.
    """
    T, B = E.shape
    start = np.maximum(np.arange(T) - W + 1, 0)
    alpha = E[start].copy()
    alpha /= alpha.sum(axis=1, keepdims=True)
    for k in range(1, W):
        pos = np.minimum(start + k, np.arange(T))
        alpha = (alpha @ A) * E[pos]
        alpha /= np.maximum(alpha.sum(axis=1, keepdims=True), 1e-300)
    return _readout(alpha, freqs), np.full(T, W, dtype=int)


def policy_deferred(E: np.ndarray, A: np.ndarray, freqs: np.ndarray,
                    threshold: float, max_depth: int = MAX_DEPTH
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    QEPC deferred evaluation.

    The belief starts from the local emission alone. While its normalized
    entropy exceeds `threshold`, one earlier window is recruited and its
    evidence propagated forward. Collapse occurs when the posterior resolves
    or the depth limit is reached, so |S| is data-dependent.

    Returns (estimate, |S| per query, final posterior entropy).
    """
    T, B = E.shape
    Apow = [None, A.copy()]
    for k in range(2, max_depth + 1):
        Apow.append(Apow[-1] @ A)

    belief = E.copy()
    belief /= belief.sum(axis=1, keepdims=True)
    S = np.ones(T, dtype=int)
    done = np.zeros(T, dtype=bool)

    for depth in range(1, max_depth):
        H = _norm_entropy(belief)
        done |= (~done) & (H < threshold)
        active = (~done) & (np.arange(T) >= depth)
        if not active.any():
            break
        prop = E[np.arange(T)[active] - depth] @ Apow[depth]
        prop /= np.maximum(prop.sum(axis=1, keepdims=True), 1e-300)
        belief[active] *= prop
        belief[active] /= np.maximum(belief[active].sum(axis=1, keepdims=True), 1e-300)
        S[active] = depth + 1
        done |= (~done) & (np.arange(T) < depth)

    return _readout(belief, freqs), S, _norm_entropy(belief)


def policy_fixedlag(E: np.ndarray, A: np.ndarray, freqs: np.ndarray,
                    W: int = W_BASELINE, L: int = 2
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Causal filter with L windows of lookahead. NOT a matched comparison."""
    T, B = E.shape
    est_alpha, _ = None, None
    start = np.maximum(np.arange(T) - W + 1, 0)
    alpha = E[start].copy()
    alpha /= alpha.sum(axis=1, keepdims=True)
    for k in range(1, W):
        pos = np.minimum(start + k, np.arange(T))
        alpha = (alpha @ A) * E[pos]
        alpha /= np.maximum(alpha.sum(axis=1, keepdims=True), 1e-300)
    if L == 0:
        return _readout(alpha, freqs), np.full(T, W, dtype=int)

    beta = np.ones((T, B)) / B
    for k in range(L, 0, -1):
        pos = np.minimum(np.arange(T) + k, T - 1)
        beta = (A @ (E[pos] * beta).T).T
        beta /= np.maximum(beta.sum(axis=1, keepdims=True), 1e-300)
    g = alpha * beta
    g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
    return _readout(g, freqs), np.full(T, W + L, dtype=int)


def policy_smoother(E: np.ndarray, A: np.ndarray, freqs: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full forward-backward. Non-causal; reference only."""
    T, B = E.shape
    al = np.zeros((T, B))
    be = np.zeros((T, B))
    al[0] = E[0] / E[0].sum()
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * E[t]
        s = al[t].sum()
        al[t] = al[t] / s if s > 0 else np.full(B, 1.0 / B)
    be[-1] = 1.0 / B
    for t in range(T - 2, -1, -1):
        be[t] = A @ (E[t + 1] * be[t + 1])
        s = be[t].sum()
        be[t] = be[t] / s if s > 0 else np.full(B, 1.0 / B)
    g = al * be
    g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
    return _readout(g, freqs), np.full(T, T, dtype=int), _norm_entropy(g)


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score(est: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    err = np.abs(est - truth)
    return {
        'MAE': float(err.mean()),
        'median': float(np.median(err)),
        'le5': float((err <= 5).mean() * 100),
        'le10': float((err <= 10).mean() * 100),
    }


def _fmt(name, s, extra=''):
    return (f'  {name:<34}{s["MAE"]:>7.2f}{s["median"]:>8.2f}'
            f'{s["le5"]:>8.1f}%{extra}')


# ─── LOSO harness ────────────────────────────────────────────────────────────

def _precompute(ds: Dataset, sigma: float, alpha: float):
    A = transition(ds.freqs, sigma)
    per = {}
    for s in ds.subjects:
        m = ds.mask(s)
        per[s] = (emission(ds.spec[m], alpha), ds.hr[m])
    return A, per


def run_loso(ds: Dataset, policy: str, W: int = W_BASELINE,
             threshold: float = 0.40, L: int = 2,
             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-subject-out evaluation of a single policy.
    Returns (pooled |error|, pooled |S|).
    """
    grid = list(itertools.product(SIGMA_GRID, ALPHA_GRID))
    cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}

    for sigma, alpha in grid:
        A, per = _precompute(ds, sigma, alpha)
        for s in ds.subjects:
            E, hr = per[s]
            if policy == 'naive':
                est, S = policy_naive(ds.spec[ds.mask(s)], ds.freqs)
            elif policy == 'causal':
                est, S = policy_causal(E, A, ds.freqs, W)
            elif policy == 'deferred':
                est, S, _ = policy_deferred(E, A, ds.freqs, threshold)
            elif policy == 'fixedlag':
                est, S = policy_fixedlag(E, A, ds.freqs, W, L)
            elif policy == 'smoother':
                est, S, _ = policy_smoother(E, A, ds.freqs)
            else:
                raise ValueError(policy)
            cache[(sigma, alpha, s)] = (np.abs(est - hr), S)

    errs, Ss = [], []
    for held in ds.subjects:
        best = None
        for sigma, alpha in grid:
            e = np.concatenate([cache[(sigma, alpha, s)][0]
                                for s in ds.subjects if s != held])
            if best is None or e.mean() < best[0]:
                best = (e.mean(), sigma, alpha)
        _, sigma, alpha = best
        err, S = cache[(sigma, alpha, held)]
        errs.append(err)
        Ss.append(S)
    return np.concatenate(errs), np.concatenate(Ss)


# ─── Reports ─────────────────────────────────────────────────────────────────

def report_table_v(ds: Dataset, export_S: Optional[str] = None):
    print('\nTABLE V  Accuracy, PPG-DaLiA heart-rate tracking '
          '(15 subjects, leave-one-subject-out)')
    print(f'  {"Method":<34}{"MAE":>7}{"median":>8}{"<=5 BPM":>9}{"  mean |S|"}')
    print('  ' + '-' * 62)

    saved = {}
    err, S = run_loso(ds, 'naive')
    print(_fmt('Naive argmax (no inference)', score_from(err), f'{S.mean():>10.2f}'))

    err, S = run_loso(ds, 'causal')
    print(_fmt(f'Causal fixed-window (W={W_BASELINE})', score_from(err),
               f'{S.mean():>10.2f}'))

    for th in THRESHOLDS:
        err, S = run_loso(ds, 'deferred', threshold=th)
        print(_fmt(f'QEPC deferred (H<{th:.2f})', score_from(err),
                   f'{S.mean():>10.2f}'))
        saved[f'S_{int(th*100)}'] = S

    print('  ' + '-' * 62)
    err, S = run_loso(ds, 'fixedlag')
    print(_fmt('Fixed-lag smoother (L=2)*', score_from(err), f'{S.mean():>10.2f}'))
    err, S = run_loso(ds, 'smoother')
    print(_fmt('Non-causal smoother*', score_from(err), '       all'))
    print('  *not matched: requires future observations')

    if export_S and saved:
        np.savez_compressed(export_S, **saved)
        print(f'\n  |S| distributions written to {export_S}')


def score_from(err: np.ndarray) -> Dict[str, float]:
    return {'MAE': float(err.mean()), 'median': float(np.median(err)),
            'le5': float((err <= 5).mean() * 100),
            'le10': float((err <= 10).mean() * 100)}


def report_gating(ds: Dataset, sigma=0.04, alpha=2.0):
    """Confidence-gated operation (manuscript Section VI-H)."""
    A, per = _precompute(ds, sigma, alpha)
    est_all, ent_all, hr_all = [], [], []
    for s in ds.subjects:
        E, hr = per[s]
        est, _, ent = policy_smoother(E, A, ds.freqs)
        est_all.append(est); ent_all.append(ent); hr_all.append(hr)
    est = np.concatenate(est_all)
    ent = np.concatenate(ent_all)
    hr = np.concatenate(hr_all)
    err = np.abs(est - hr)

    try:
        from scipy.stats import spearmanr
        rho = spearmanr(ent, err).statistic
    except Exception:
        rho = float('nan')

    print('\nCONFIDENCE-GATED OPERATION  (Section VI-H)')
    print(f'  Spearman rho(entropy, |error|) = {rho:.3f}')
    print(f'  {"coverage":>10}{"MAE":>9}{"median":>9}{"<=5 BPM":>10}')
    for q in (1.0, 0.8, 0.6, 0.5):
        th = np.quantile(ent, q)
        k = ent <= th
        e = err[k]
        print(f'  {100*k.mean():>9.0f}%{e.mean():>9.2f}{np.median(e):>9.2f}'
              f'{100*(e<=5).mean():>9.1f}%')


def report_pmf_width(ds: Dataset, sigma=0.04, alpha=2.0):
    """Accuracy versus PMF width (manuscript Section VI-B, SD-1 Table 4.2)."""
    print('\nACCURACY vs PMF WIDTH  (non-causal smoother, isolates width effect)')
    print(f'  {"states":>7}{"bits/node":>11}{"MAE":>8}{"median":>9}{"<=5 BPM":>10}')
    B0 = ds.spec.shape[1]
    for nb in (96, 48, 32, 16, 8, 4):
        if nb == B0:
            spec, freqs = ds.spec, ds.freqs
        else:
            edges = np.linspace(0, B0, nb + 1).astype(int)
            spec = np.stack([ds.spec[:, edges[i]:edges[i+1]].max(axis=1)
                             for i in range(nb)], axis=1)
            freqs = np.array([ds.freqs[edges[i]:edges[i+1]].mean()
                              for i in range(nb)])
        sg = max(sigma, (freqs[1] - freqs[0]) * 1.3)
        A = transition(freqs, sg)
        est_all, hr_all = [], []
        for s in ds.subjects:
            m = ds.mask(s)
            E = emission(spec[m], alpha)
            est, _, _ = policy_smoother(E, A, freqs)
            est_all.append(est); hr_all.append(ds.hr[m])
        err = np.abs(np.concatenate(est_all) - np.concatenate(hr_all))
        print(f'  {nb:>7}{nb*16:>11}{err.mean():>8.2f}'
              f'{np.median(err):>9.2f}{100*(err<=5).mean():>9.1f}%')
    print('  note: the 4-state row is an artifact of 43.5 BPM bins, not a '
          'usable operating point (see SD-1 Section 4.2)')


def report_coverage(ds: Dataset):
    """Spectral coverage check that motivated the full-spectrum emission."""
    step = ds.freqs[1] - ds.freqs[0]
    tb = np.clip(np.round((ds.hr / 60 - ds.freqs[0]) / step).astype(int),
                 0, len(ds.freqs) - 1)
    i = np.arange(len(ds.hr))
    mag = ds.spec[i, tb]
    is_max = (mag >= ds.spec.max(axis=1) - 1e-9)
    print('\nSPECTRAL COVERAGE  (Section VI-B)')
    print(f'  median magnitude at true-HR bin : {np.median(mag):.3f}')
    print(f'  fraction with magnitude > 0.2   : {100*(mag>0.2).mean():.1f}%')
    print(f'  fraction where it is the maximum: {100*is_max.mean():.1f}%')
    print('  -> peak selection, not peak detection')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description='QEPC PPG-DaLiA benchmark')
    ap.add_argument('--npz', required=True, help='feature file from preprocess_ppg_dalia_v5.py')
    ap.add_argument('--loso', action='store_true', help='Table V accuracy (default)')
    ap.add_argument('--gating', action='store_true', help='confidence-gated operation')
    ap.add_argument('--pmf-width', action='store_true', help='accuracy vs PMF width')
    ap.add_argument('--coverage', action='store_true', help='spectral coverage check')
    ap.add_argument('--all', action='store_true', help='run everything')
    ap.add_argument('--export-S', default=None, help='write |S| arrays to .npz')
    args = ap.parse_args()

    ds = load(args.npz)
    print(f'loaded {len(ds.hr):,} windows, {len(ds.subjects)} subjects, '
          f'{ds.spec.shape[1]} spectral bins')

    ran = False
    if args.coverage or args.all:
        report_coverage(ds); ran = True
    if args.pmf_width or args.all:
        report_pmf_width(ds); ran = True
    if args.gating or args.all:
        report_gating(ds); ran = True
    if args.loso or args.all or not ran:
        report_table_v(ds, export_S=args.export_S)
    return 0


if __name__ == '__main__':
    sys.exit(main())
