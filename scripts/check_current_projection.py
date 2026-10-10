"""
Standalone verification script: check the current-projection (Helmholtz
split) against an analytically constructed test current with KNOWN
transverse and longitudinal parts. Run directly:

    python scripts/check_current_projection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from fields_longitudinal import build_mode_poisson_solver
from current_projection import project_transverse_current


def main():
    nx = 256
    Lx = 256.0
    dx = Lx / nx
    eps_0 = 0.04
    x = np.arange(nx) * dx
    q = 2.0 * np.pi / Lx

    ky = 2.0 * np.pi * 1 / 256.0   # mode (m=1, n=2), arbitrary but ky != kz
    kz = 2.0 * np.pi * 2 / 256.0
    kperp2 = ky ** 2 + kz ** 2

    # --- known transverse part: Jx_T=0, Jy_T=A sin(qx), Jz_T=-(ky/kz) Jy_T ---
    A = 2.0
    JxT_known = np.zeros(nx, dtype=complex)
    JyT_known = A * np.sin(q * x).astype(complex)
    JzT_known = -(ky / kz) * JyT_known

    # --- known longitudinal part: J_L = grad(psi_test), psi_test = B sin(qx) ---
    B = 1.5
    psi_test = B * np.sin(q * x)
    JxL_known = (B * q * np.cos(q * x)).astype(complex)
    JyL_known = 1j * ky * psi_test
    JzL_known = 1j * kz * psi_test

    Jx = JxT_known + JxL_known
    Jy = JyT_known + JyL_known
    Jz = JzT_known + JzL_known

    solver = build_mode_poisson_solver(nx, dx, kperp2, eps_0)
    JxT_computed, JyT_computed, JzT_computed = project_transverse_current(
        Jx, Jy, Jz, ky, kz, dx, solver)

    def rel_err(a, b):
        denom = np.max(np.abs(b)) if np.max(np.abs(b)) > 0 else 1.0
        return np.max(np.abs(a - b)) / denom

    print('=== Current projection verification ===')
    print(f'  ky={ky:.6f}  kz={kz:.6f}  kperp2={kperp2:.6f}')
    print(f'  rel err Jx_T: {rel_err(JxT_computed, JxT_known):.4%}  (known max|JxT|={np.max(np.abs(JxT_known)):.4g})')
    print(f'  rel err Jy_T: {rel_err(JyT_computed, JyT_known):.4%}')
    print(f'  rel err Jz_T: {rel_err(JzT_computed, JzT_known):.4%}')

    # sanity: computed J_T should be divergence-free
    dJxT_dx = (np.roll(JxT_computed, -1) - np.roll(JxT_computed, 1)) / (2 * dx)
    div_JT_computed = dJxT_dx + 1j * ky * JyT_computed + 1j * kz * JzT_computed
    print(f'  max|div(J_T_computed)| = {np.max(np.abs(div_JT_computed)):.4g} (should be ~0)')


if __name__ == '__main__':
    main()
