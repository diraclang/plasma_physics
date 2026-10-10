"""
Standalone verification script: check that the double Harris sheet + guide
field equilibrium satisfies (approximate) force balance, and plot the
profiles for a visual sanity check. Run directly:

    python scripts/check_equilibrium.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SimulationConfig, ModeSpec
from equilibrium import Bz_eq, By_eq, n_eq, dBz_dx_eq, derive_units, drift_velocities


def main():
    cfg = SimulationConfig(
        Lx=256.0, Ly=256.0, Lz=256.0, lam=20.0, B0=4.0, Bg=4.0, n_bg=0.2,
        mass_ratio=4.0, c=5.0, V_Te=2.0, T_i_over_T_e=1.0,
        modes=[ModeSpec(m=0, n=1), ModeSpec(m=1, n=1)],
    )
    units = derive_units(cfg)

    x = np.linspace(0.0, cfg.Lx, 2000)
    Bz = Bz_eq(x, cfg)
    By = By_eq(x, cfg)
    n = n_eq(x, cfg)

    v_iy, v_ey, T_i, T_e = drift_velocities(cfg, units)
    T_total = T_i + T_e

    # Total pressure: kinetic (n * T_total, isotropic-temperature approx) +
    # magnetic pressure from the REVERSING component only (guide field By is
    # uniform in x so it contributes a constant offset and doesn't affect
    # whether P(x) is flat).
    P_kinetic = n * T_total
    P_mag = Bz ** 2 / (2.0 * units.mu_0)
    P_total = P_kinetic + P_mag

    print('=== Derived units ===')
    print(f'  eps_0={units.eps_0:.6g}  mu_0={units.mu_0:.6g}  c={units.c:.6g}')
    print(f'  m_i={units.m_i:.6g}  m_e={units.m_e:.6g}  e={units.e:.6g}')
    print(f'  omega_pi={units.omega_pi:.6g}  omega_pe={units.omega_pe:.6g}  d_i={units.d_i:.6g}')
    print(f'  Omega_ci={units.Omega_ci:.6g}  Omega_ce={units.Omega_ce:.6g}  v_A={units.v_A:.6g}')
    print()
    print('=== Equilibrium sanity checks ===')
    print(f'  T_i={T_i:.6g}  T_e={T_e:.6g}  V_Ti={(T_i/units.m_i)**0.5:.6g} (should be 1)  '
          f'V_Te={(T_e/units.m_e)**0.5:.6g} (should be 2)')
    print(f'  v_iy={v_iy:.6g}  v_ey={v_ey:.6g}  (drift speeds, sheet population)')
    print(f'  max|Bz|={np.max(np.abs(Bz)):.6g} (should be ~{cfg.B0:.3g} at sheet centers)')
    print(f'  max n={np.max(n):.6g} (should be ~1.0 + n_bg={1.0+cfg.n_bg:.3g} at sheet centers)')
    print(f'  min n={np.min(n):.6g} (should be ~n_bg={cfg.n_bg:.3g} far from sheets)')
    print()
    print('=== Force balance: P_total(x) = n(x)*(Ti+Te) + Bz(x)^2/(2 mu_0) ===')
    print(f'  mean(P_total) = {np.mean(P_total):.6g}')
    print(f'  std(P_total)  = {np.std(P_total):.6g}')
    print(f'  relative std  = {np.std(P_total)/np.mean(P_total):.4%}')
    print('  (P_total is exactly constant analytically: Bz^2 = B0^2 - B0^2*sech^2 by'
          ' the tanh/sech identity, so the sech^2 terms cancel between magnetic and'
          ' kinetic pressure regardless of n_bg, which just adds a constant offset)')

    # resonant surfaces for the configured modes
    print()
    print('=== Resonant surfaces for configured modes ===')
    for mode in cfg.modes:
        roots = cfg.resonant_x(mode)
        ky, kz = cfg.k_y(mode), cfg.k_z(mode)
        print(f'  mode (m={mode.m}, n={mode.n})  ky={ky:.4f} kz={kz:.4f}  '
              f'resonant x = {["%.3f" % r for r in roots]}')

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(x, Bz, label='Bz(x)')
    axes[0].plot(x, By, label='By(x) = Bg', linestyle='--')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_ylabel('B (units of B0)')
    axes[0].legend()

    axes[1].plot(x, n, color='tab:green')
    axes[1].set_ylabel('n (units of n0)')

    axes[2].plot(x, P_total, color='tab:red', label='P_total')
    axes[2].axhline(np.mean(P_total), color='gray', linestyle=':', label='mean')
    axes[2].set_ylabel('P_total')
    axes[2].set_xlabel('x (units of d_i)')
    axes[2].legend()

    for mode in cfg.modes:
        for r in cfg.resonant_x(mode):
            for ax in axes:
                ax.axvline(r, color='purple', alpha=0.3, linewidth=1)

    fig.suptitle('Double Harris sheet equilibrium + guide field')
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'equilibrium_check.png'
    fig.savefig(out, dpi=150)
    print()
    print('Saved plot:', out)


if __name__ == '__main__':
    main()
