"""
energy_tables.py
================
Reproduces the energy results of the manuscript from the measured
active-subgraph distributions produced by ppg_benchmark.py.

Generates:
  Table V, energy columns   - per-query energy and ratio versus the baseline
  Section VI-F              - retention technology comparison
  Section VI-G              - per-query energy composition and sensitivity
  Table III                 - provisioning and thermal headroom

USAGE
-----
    python ppg_benchmark.py --npz ppg_dalia_v5.npz --loso --export-S S_loso.npz
    python energy_tables.py --S S_loso.npz
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import numpy as np

from energy_model_v2 import HW, EnergyModelV2

W_BASELINE = 16.0
CADENCE_HZ = 0.5
LAYERS = {'L5 Cerebellum (4 nm)': 5, 'L4 Cortex (40 nm RRAM)': 50,
          'L3 Thalamus (7 nm)': 10, 'L2 Hippocampus (22 nm)': 20,
          'L1 Pons (22 nm eMRAM)': 30}
RTH = {'L4 Cortex (40 nm RRAM)': 1.2, 'L3 Thalamus (7 nm)': 1.0,
       'L2 Hippocampus (22 nm)': 1.0, 'L1 Pons (22 nm eMRAM)': 0.8}
EMRAM_CELL_UM2 = 0.046      # TSMC 22ULL 1-bit cell


def retained_for(S_mean: float, cap: float) -> float:
    """Nodes whose PMF must be retained between queries."""
    return float(min(cap, np.ceil(S_mean * 2)))


def table_v_energy(S: Dict[str, np.ndarray], hw: HW):
    m = EnergyModelV2(hw, retention='emram')
    base = m.query_energy(W_BASELINE, n_retained=W_BASELINE)
    print('\nTABLE V  energy columns  (eMRAM retention)')
    print(f'  {"Method":<30}{"mean |S|":>10}{"E/query":>12}{"vs base":>10}')
    print('  ' + '-' * 62)
    print(f'  {"Causal fixed-window (W=16)":<30}{W_BASELINE:>10.2f}'
          f'{base.total*1e9:>11.2f}n{"1.00x":>10}')
    out = {}
    for key in sorted(S):
        s = float(S[key].mean())
        th = int(key.split('_')[1]) / 100
        q = m.query_energy(s, n_retained=retained_for(s, hw.retain_depth))
        out[th] = (s, q)
        print(f'  {f"QEPC deferred (H<{th:.2f})":<30}{s:>10.2f}'
              f'{q.total*1e9:>11.2f}n{base.total/q.total:>9.2f}x')
    return base, out


def retention_comparison(S: Dict[str, np.ndarray], hw: HW, headline=0.30):
    key = f'S_{int(headline*100)}'
    if key not in S:
        key = sorted(S)[0]
    s = float(S[key].mean())
    nret = retained_for(s, hw.retain_depth)

    me = EnergyModelV2(hw, 'emram')
    ms = EnergyModelV2(hw, 'sram')
    qe, be = me.query_energy(s, nret), me.query_energy(W_BASELINE, n_retained=W_BASELINE)
    qs, bs = ms.query_energy(s, nret), ms.query_energy(W_BASELINE, n_retained=W_BASELINE)

    print('\nSECTION VI-F  retention technology')
    print(f'  {"Retention":<26}{"QEPC":>12}{"baseline":>12}{"ratio":>10}')
    print('  ' + '-' * 60)
    print(f'  {"Volatile (SRAM)":<26}{qs.total*1e6:>11.2f}u{bs.total*1e6:>11.2f}u'
          f'{bs.total/qs.total:>9.2f}x   <- net loss')
    print(f'  {"Non-volatile (eMRAM)":<26}{qe.total*1e9:>11.2f}n{be.total*1e9:>11.2f}n'
          f'{be.total/qe.total:>9.2f}x')
    print(f'  gap between technologies: {qs.total/qe.total:,.0f}x')
    return qe


def composition(q):
    print('\nSECTION VI-G  per-query energy composition')
    for lbl, v in (('State-store access', q.E_memory), ('Inter-layer TSV', q.E_tsv),
                   ('Controller', q.E_ctrl), ('Compute', q.E_compute),
                   ('Adiabatic erasure', q.E_erase)):
        print(f'  {lbl:<24}{100*v/q.total:>7.2f}%')


def sensitivity(S: Dict[str, np.ndarray], hw: HW, headline=0.30):
    key = f'S_{int(headline*100)}'
    if key not in S:
        key = sorted(S)[0]
    s = float(S[key].mean())
    nret = retained_for(s, hw.retain_depth)
    print('\nSENSITIVITY  (range of total energy across published uncertainty)')
    print(f'  {"Parameter":<30}{"low":>10}{"nominal":>10}{"high":>10}{"span":>9}')
    print('  ' + '-' * 70)
    sweeps = [('eMRAM write per bit', 'E_mram_write_bit', 100e-15, 1e-12),
              ('Inter-layer transfer per bit', 'E_tsv_bit', 10e-15, 200e-15),
              ('Peripheral overhead factor', 'periph_factor', 1.5, 3.0),
              ('Controller per query', 'E_ctrl_query', 20e-12, 500e-12),
              ('RRAM CIM energy per MAC', 'E_RRAM_mac', 5e-15, 50e-15)]
    for name, attr, lo, hi in sweeps:
        vals = []
        for v in (lo, getattr(HW(), attr), hi):
            h = HW(); setattr(h, attr, v); h.__post_init__()
            vals.append(EnergyModelV2(h, 'emram').query_energy(s, nret).total * 1e9)
        print(f'  {name:<30}{vals[0]:>9.2f}n{vals[1]:>9.2f}n{vals[2]:>9.2f}n'
              f'{vals[2]/vals[0]:>8.2f}x')


def table_iii(q, hw: HW):
    E = q.total
    P = E * CADENCE_HZ
    total_area = sum(LAYERS.values())
    bits = hw.retain_depth * hw.pmf_bits
    used_mm2 = bits * EMRAM_CELL_UM2 / 1e6

    print('\nTABLE III  provisioning and thermal headroom')
    print(f'  active inference power : {P*1e9:.2f} nW  ({E*1e9:.2f} nJ/query '
          f'at {CADENCE_HZ} Hz)')
    print(f'  power density          : {P/(total_area*1e-2):.2e} W/cm2')
    print(f'  retained state         : {bits:,} bits = {bits/8/1024:.1f} kB '
          f'= {used_mm2:.5f} mm2')
    print(f'  L1 utilisation         : {100*used_mm2/LAYERS["L1 Pons (22 nm eMRAM)"]:.4f}%')
    print(f'\n  {"Layer":<26}{"area":>8}{"R_th":>7}{"dT":>14}')
    for name, area in LAYERS.items():
        r = RTH.get(name)
        dt = f'{P*r/area:.2e} C' if r else '-- (heat sink)'
        print(f'  {name:<26}{area:>6} mm2{(f"{r:.1f}" if r else "--"):>7}{dt:>14}')
    print(f'\n  headroom  {"envelope":>10}{"queries/s":>14}{"channels":>14}')
    for mw in (1, 10, 100):
        n = mw * 1e-3 / E
        print(f'            {mw:>8} mW{n:>14,.0f}{n/CADENCE_HZ:>14,.0f}')


def main() -> int:
    ap = argparse.ArgumentParser(description='QEPC energy tables')
    ap.add_argument('--S', required=True,
                    help='.npz of |S| arrays from ppg_benchmark.py --export-S')
    ap.add_argument('--headline', type=float, default=0.30,
                    help='threshold used for Table III and composition')
    args = ap.parse_args()

    S = dict(np.load(args.S))
    hw = HW()
    print(f'PMF width {hw.n_states} states ({hw.pmf_bits} b/node), '
          f'tau = {hw.tau_query} s, retention depth {hw.retain_depth}')

    base, out = table_v_energy(S, hw)
    q = retention_comparison(S, hw, args.headline)
    composition(q)
    sensitivity(S, hw, args.headline)
    table_iii(q, hw)
    return 0


if __name__ == '__main__':
    sys.exit(main())
