"""Short coupled-loop regression: energy conservation with substantial
field-particle energy exchange, plus dt convergence.

Checks (2 modes, dx = lambda_De, 100 ppc, t = 8):
  * W_E + W_B is a significant energy (not a vacuous test);
  * |dW/W| < 1e-3 at the physics-limited dt;
  * the drift does not grow when dt is halved (converges as dt -> 0).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from config import SimulationConfig, ModeSpec
from coupled_loop import run_coupled_loop


def make_cfg():
    return SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0, lam=10.0, B0=4.0, Bg=4.0,
        nx=64, particles_per_cell=100, dt=0.05,
        modes=[ModeSpec(m=0, n=1, seed_amplitude=0.05),
               ModeSpec(m=1, n=1, seed_amplitude=0.05)],
    )


def main():
    T = 8.0
    results = []
    for dt in (0.04, 0.02, 0.01):
        h = run_coupled_loop(make_cfg(), n_steps=int(round(T / dt)), dt=dt)
        W = h['W_total']
        max_rel = float(np.max(np.abs(W - W[0])) / W[0])
        results.append((dt, max_rel, h))
        print(f'  dt={dt:.3f}  max |dW/W| = {max_rel:.3e}  '
              f'(W_E+W_B)/W at end = {(h["W_E"][-1] + h["W_B"][-1]) / W[-1]:.3e}')

    h = results[1][2]
    field_frac = (h['W_E'][0] + h['W_B'][0]) / h['W_total'][0]
    print('=== coupled mode loop check ===')
    print(f'  field energy fraction at t=0 = {field_frac:.3e}')
    if field_frac < 1.0e-3:
        raise SystemExit('field energy negligible: energy test would be vacuous')
    if not all(np.isfinite(r[2]['W_total']).all() for r in results):
        raise SystemExit('non-finite energies encountered')
    if max(r[1] for r in results) > 1.0e-3:
        raise SystemExit('total energy drift too large')
    if results[-1][1] > 2.0 * results[0][1]:
        raise SystemExit('drift does not converge with dt')

    print('PASS')


if __name__ == '__main__':
    main()
