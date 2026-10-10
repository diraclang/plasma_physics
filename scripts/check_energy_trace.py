"""Long energy trace with the full retained mode bank at the physics dt.

Modes m in [-3,3], n in [-1,1] (folded to 10 real half-space modes), dx =
lambda_De, 100 ppc, t = 22 (about 500 steps at dt ~ 0.044, ~8 electron
gyro-periods).  Reports the energy partition and drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from config import SimulationConfig, mode_range
from coupled_loop import run_coupled_loop


def main():
    cfg = SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0, lam=10.0, B0=4.0, Bg=4.0,
        nx=64, particles_per_cell=100, dt=0.05,
        modes=mode_range(range(-3, 4), range(-1, 2), seed_amplitude=0.02),
    )
    h = run_coupled_loop(cfg, n_steps=500)
    W = h['W_total']
    rel = (W - W[0]) / W[0]
    max_rel = float(np.max(np.abs(rel)))

    print('=== long energy trace ===')
    print(f'  dt = {h["dt"]:.4f}, steps = {len(W)}, t_end = {h["t"][-1]:.1f}, '
          f'half-space modes = {len(h["modes"])}')
    print(f'  W_K  : {h["W_K"][0]:.6e} -> {h["W_K"][-1]:.6e}')
    print(f'  W_E  : {h["W_E"][0]:.6e} -> {h["W_E"][-1]:.6e}')
    print(f'  W_B  : {h["W_B"][0]:.6e} -> {h["W_B"][-1]:.6e}')
    print(f'  final relative drift = {rel[-1]:+.3e}')
    print(f'  max relative drift   = {max_rel:.3e}')

    if not np.all(np.isfinite(W)):
        raise SystemExit('non-finite energies encountered')
    if max_rel > 5.0e-4:
        raise SystemExit(f'total energy drift too large: {max_rel:.3e}')

    print('PASS')


if __name__ == '__main__':
    main()
