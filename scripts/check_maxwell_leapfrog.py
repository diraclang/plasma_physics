"""
Standalone verification script: launch a vacuum (J_T=0) light wave in the
transverse Maxwell leapfrog and check it propagates at exactly speed c, with
no numerical damping or dispersion error at the resolved wavelength. Run
directly:

    python scripts/check_maxwell_leapfrog.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from fields_transverse import TransverseModeState, leapfrog_step


def main():
    nx = 256
    Lx = 256.0
    dx = Lx / nx
    x = np.arange(nx) * dx
    c = 5.0
    eps_0 = 0.04

    # mode (ky=kz=0): pure 1D TE wave along x, polarized E_y / B_z, moving in +x
    ky, kz = 0.0, 0.0

    kx = 2.0 * np.pi * 4 / Lx   # 4 wavelengths across the periodic box
    E0 = 1.0

    ETy0 = (E0 * np.sin(kx * x)).astype(complex)
    Bz0 = (E0 / c * np.sin(kx * x)).astype(complex)   # right-moving wave relation E=cB
    zeros = np.zeros(nx, dtype=complex)

    state = TransverseModeState(ky=ky, kz=kz, ETx=zeros.copy(), ETy=ETy0.copy(),
                                 ETz=zeros.copy(), Bx=zeros.copy(), By=zeros.copy(),
                                 Bz=Bz0.copy())

    # CFL: dt must satisfy c*dt/dx < 1 (1D von Neumann stability bound for this scheme)
    dt = 0.4 * dx / c
    n_steps = 200
    JT = zeros.copy()

    for _ in range(n_steps):
        leapfrog_step(state, JT, JT, JT, dx, dt, c, eps_0)

    t_final = n_steps * dt
    shift = c * t_final
    ETy_analytic = E0 * np.sin(kx * (x - shift))

    err = np.max(np.abs(state.ETy.real - ETy_analytic)) / E0
    amp_measured = np.max(np.abs(state.ETy.real))

    print('=== Vacuum light wave propagation check ===')
    print(f'  nx={nx}  dx={dx}  dt={dt:.6g}  c={c}  CFL=c*dt/dx={c*dt/dx:.3f}')
    print(f'  wavelength = {2*np.pi/kx:.3f} (lambda_De), {n_steps} steps, t_final={t_final:.4f}')
    print(f'  expected propagation distance = c*t_final = {shift:.4f}')
    print(f'  measured wave amplitude: {amp_measured:.6f} (expected {E0}, checks no damping/growth)')
    print(f'  max|ETy_numeric - ETy_analytic(shifted)| / E0 = {err:.4%}')

    print()
    print('=== Convergence check: error should shrink as nx (and dt) increase ===')
    for nx_test in [64, 128, 256, 512]:
        dx_t = Lx / nx_test
        x_t = np.arange(nx_test) * dx_t
        ETy0_t = (E0 * np.sin(kx * x_t)).astype(complex)
        Bz0_t = (E0 / c * np.sin(kx * x_t)).astype(complex)
        zeros_t = np.zeros(nx_test, dtype=complex)
        state_t = TransverseModeState(ky=0.0, kz=0.0, ETx=zeros_t.copy(), ETy=ETy0_t.copy(),
                                       ETz=zeros_t.copy(), Bx=zeros_t.copy(), By=zeros_t.copy(),
                                       Bz=Bz0_t.copy())
        dt_t = 0.4 * dx_t / c
        n_steps_t = int(round(t_final / dt_t))
        for _ in range(n_steps_t):
            leapfrog_step(state_t, zeros_t, zeros_t, zeros_t, dx_t, dt_t, c, eps_0)
        shift_t = c * n_steps_t * dt_t
        ETy_analytic_t = E0 * np.sin(kx * (x_t - shift_t))
        err_t = np.max(np.abs(state_t.ETy.real - ETy_analytic_t)) / E0
        print(f'  nx={nx_test:4d}  dt={dt_t:.6g}  steps={n_steps_t:5d}  relative error = {err_t:.4%}')


if __name__ == '__main__':
    main()
