"""
Standalone verification script: check the per-mode Poisson solver against a
known analytic solution. Run directly:

    python scripts/check_poisson.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from fields_longitudinal import build_mode_poisson_solver


def analytic_case(nx, Lx, kperp2, eps_0, A=1.3):
    """rho_k(x) = A*sin(q*x), q = 2*pi/Lx (fits exactly one period in the
    periodic domain). Analytic solution of phi'' - kperp2*phi = -rho/eps_0:
    phi(x) = A / (eps_0 * (q^2 + kperp2)) * sin(q*x)."""
    dx = Lx / nx
    x = np.arange(nx) * dx
    q = 2.0 * np.pi / Lx
    rho = A * np.sin(q * x)
    phi_analytic = A / (eps_0 * (q ** 2 + kperp2)) * np.sin(q * x)
    return x, dx, rho.astype(complex), phi_analytic


def main():
    nx = 128
    Lx = 256.0
    eps_0 = 0.04   # matches the derived value from the 1993 parameters

    print('=== Non-degenerate case (kperp2 > 0) ===')
    kperp2 = 0.05
    x, dx, rho, phi_analytic = analytic_case(nx, Lx, kperp2, eps_0)
    solver = build_mode_poisson_solver(nx, dx, kperp2, eps_0)
    phi_numeric = solver.solve(rho)
    err = np.max(np.abs(phi_numeric.real - phi_analytic)) / np.max(np.abs(phi_analytic))
    print(f'  kperp2={kperp2}, nx={nx}, Lx={Lx}')
    print(f'  max|phi_numeric - phi_analytic| / max|phi_analytic| = {err:.3%}')
    print(f'  sample: phi_analytic[10]={phi_analytic[10]:.6g}  phi_numeric[10]={phi_numeric[10].real:.6g}')

    print()
    print('=== Degenerate case (kperp2 == 0, the m=n=0 "mode") ===')
    kperp2 = 0.0
    x, dx, rho, phi_analytic = analytic_case(nx, Lx, kperp2, eps_0)
    solver = build_mode_poisson_solver(nx, dx, kperp2, eps_0)
    phi_numeric = solver.solve(rho)
    # both solutions have zero mean by construction (sin over a full period);
    # re-center analytic too, just in case of float round-off
    phi_analytic_centered = phi_analytic - np.mean(phi_analytic)
    err = np.max(np.abs(phi_numeric.real - phi_analytic_centered)) / np.max(np.abs(phi_analytic_centered))
    print(f'  kperp2={kperp2}, nx={nx}, Lx={Lx}')
    print(f'  max|phi_numeric - phi_analytic| / max|phi_analytic| = {err:.3%}')
    print(f'  mean(phi_numeric) = {np.mean(phi_numeric).real:.3e} (should be ~0, gauge choice)')

    print()
    print('=== Convergence check: error should shrink as nx increases (2nd order) ===')
    kperp2 = 0.05
    for nx_test in [32, 64, 128, 256, 512]:
        x, dx, rho, phi_analytic = analytic_case(nx_test, Lx, kperp2, eps_0)
        solver = build_mode_poisson_solver(nx_test, dx, kperp2, eps_0)
        phi_numeric = solver.solve(rho)
        err = np.max(np.abs(phi_numeric.real - phi_analytic)) / np.max(np.abs(phi_analytic))
        print(f'  nx={nx_test:4d}  relative error = {err:.6%}')


if __name__ == '__main__':
    main()
