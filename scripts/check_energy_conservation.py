"""Verification of the total-energy diagnostic for the transverse Maxwell solve.

This script checks the vacuum propagation case already used in
check_maxwell_leapfrog.py, but now in terms of the energy monitor itself:

    E_total = E_electric + E_magnetic + E_kinetic

For a closed vacuum system with no particles and no source current, the
field energy should remain nearly constant in time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from energy import field_energy, total_energy
from fields_transverse import TransverseModeState, leapfrog_step


def main():
    nx = 256
    Lx = 256.0
    dx = Lx / nx
    x = np.arange(nx) * dx
    c = 5.0
    eps_0 = 0.04

    ky, kz = 0.0, 0.0
    kx = 2.0 * np.pi * 4 / Lx
    E0 = 1.0

    ETy0 = (E0 * np.sin(kx * x)).astype(complex)
    Bz0 = (E0 / c * np.sin(kx * x)).astype(complex)
    zeros = np.zeros(nx, dtype=complex)

    state = TransverseModeState(
        ky=ky,
        kz=kz,
        ETx=zeros.copy(),
        ETy=ETy0.copy(),
        ETz=zeros.copy(),
        Bx=zeros.copy(),
        By=zeros.copy(),
        Bz=Bz0.copy(),
    )

    dt = 0.4 * dx / c
    JT = zeros.copy()
    energy0 = total_energy(state, dx, eps_0=eps_0)

    for _ in range(200):
        leapfrog_step(state, JT, JT, JT, dx, dt, c, eps_0)

    energy1 = total_energy(state, dx, eps_0=eps_0)
    rel_drift = abs(energy1 - energy0) / energy0

    print('=== energy conservation check ===')
    print(f'  E_total(t=0) = {energy0:.12e}')
    print(f'  E_total(t_final) = {energy1:.12e}')
    print(f'  relative drift = {rel_drift:.4%}')

    if rel_drift > 5.0e-3:
        raise SystemExit(f'energy drift too large: {rel_drift:.4%}')

    print('PASS')


if __name__ == '__main__':
    main()
