"""
Standalone verification script: load ion and electron macroparticles and
check that their binned moments (density, drift velocity, current) reproduce
the intended equilibrium. Run directly:

    python scripts/check_particles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SimulationConfig, ModeSpec
from equilibrium import derive_units, n_eq, dBz_dx_eq, drift_velocities
from particles import load_species


def bin_moment(x, values, weight, bin_edges):
    """Return, per bin, sum(weight*values) and sum(weight) (so mean = ratio)."""
    idx = np.digitize(x, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    sum_wv = np.zeros(n_bins)
    sum_w = np.zeros(n_bins)
    valid = (idx >= 0) & (idx < n_bins)
    np.add.at(sum_wv, idx[valid], (weight * values)[valid])
    np.add.at(sum_w, idx[valid], weight if np.isscalar(weight) else weight[valid])
    return sum_wv, sum_w


def main():
    cfg = SimulationConfig(
        Lx=256.0, Ly=256.0, Lz=256.0, lam=20.0, B0=4.0, Bg=4.0, n_bg=0.2,
        mass_ratio=4.0, c=5.0, V_Te=2.0, T_i_over_T_e=1.0,
        particles_per_cell=2000,
        modes=[ModeSpec(m=0, n=1), ModeSpec(m=1, n=1)],
    )
    units = derive_units(cfg)
    rng = np.random.default_rng(42)

    ions = load_species(cfg, units, 'ion', rng)
    electrons = load_species(cfg, units, 'electron', rng)

    print(f'Loaded {ions.n} ion macroparticles, {electrons.n} electron macroparticles')
    print(f'Ion weight={ions.weight:.6g}  q={ions.q:.6g}  m={ions.m:.6g}')
    print(f'Electron weight={electrons.weight:.6g}  q={electrons.q:.6g}  m={electrons.m:.6g}')

    n_bins = 128
    bin_edges = np.linspace(0.0, cfg.Lx, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_volume = (cfg.Lx / n_bins) * cfg.Ly * cfg.Lz

    # density check: binned particle COUNT * weight / bin_volume vs n_eq(x)
    cnt_i, _ = np.histogram(ions.x, bins=bin_edges)
    cnt_e, _ = np.histogram(electrons.x, bins=bin_edges)
    dens_i_measured = cnt_i * ions.weight / bin_volume
    dens_e_measured = cnt_e * electrons.weight / bin_volume
    dens_expected = n_eq(bin_centers, cfg)

    # vy moment per bin (mean drift), per species
    sum_wv_i, sum_w_i = bin_moment(ions.x, ions.vy, np.ones(ions.n), bin_edges)
    sum_wv_e, sum_w_e = bin_moment(electrons.x, electrons.vy, np.ones(electrons.n), bin_edges)
    mean_vy_i = np.divide(sum_wv_i, sum_w_i, out=np.full(n_bins, np.nan), where=sum_w_i > 0)
    mean_vy_e = np.divide(sum_wv_e, sum_w_e, out=np.full(n_bins, np.nan), where=sum_w_e > 0)

    # current density Jy(x) = n_i*q_i*<vy_i> + n_e*q_e*<vy_e>  (measured)
    Jy_measured = dens_i_measured * ions.q * mean_vy_i + dens_e_measured * electrons.q * mean_vy_e
    Jy_expected = -dBz_dx_eq(bin_centers, cfg) / units.mu_0

    v_iy, v_ey, T_i, T_e = drift_velocities(cfg, units)
    # restrict comparison to a window right at one sheet center where sheet
    # population dominates over background, so the expected uniform-drift
    # approximation is most valid
    center_idx = np.argmin(np.abs(bin_centers - cfg.Lx / 4.0))
    window = slice(max(0, center_idx - 3), center_idx + 4)

    print()
    print('=== Density check (near sheet center x=Lx/4) ===')
    print('  bin_center   n_i(measured)  n_e(measured)  n_eq(expected)')
    for i in range(center_idx - 2, center_idx + 3):
        print(f'  {bin_centers[i]:8.3f}   {dens_i_measured[i]:10.4f}   '
              f'{dens_e_measured[i]:10.4f}   {dens_expected[i]:10.4f}')

    print()
    print('=== Drift velocity check (near sheet center, sheet-dominated bins) ===')
    print(f'  expected: v_iy={v_iy:.6g}  v_ey={v_ey:.6g}')
    print(f'  measured (window mean): v_iy={np.nanmean(mean_vy_i[window]):.6g}  '
          f'v_ey={np.nanmean(mean_vy_e[window]):.6g}')

    print()
    print('=== Current density check Jy(x) = -(1/mu_0) dBz/dx ===')
    rel_err = np.abs(Jy_measured[window] - Jy_expected[window]) / np.abs(Jy_expected[window]).max()
    print(f'  max relative error in window: {np.max(rel_err):.3%}')
    print(f'  expected Jy at window center: {Jy_expected[center_idx]:.6g}')
    print(f'  measured Jy at window center: {Jy_measured[center_idx]:.6g}')

    print()
    print('=== Quasineutrality check: n_i - n_e should be small everywhere ===')
    charge_imbalance = dens_i_measured - dens_e_measured
    print(f'  max|n_i - n_e| = {np.max(np.abs(charge_imbalance)):.4g} '
          f'(compare to peak density ~{np.max(dens_expected):.3g})')

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(bin_centers, dens_expected, 'k-', label='n_eq(x) expected', linewidth=2)
    axes[0].plot(bin_centers, dens_i_measured, 'o', color='tab:blue', markersize=3, label='ion (measured)')
    axes[0].plot(bin_centers, dens_e_measured, 'x', color='tab:red', markersize=3, label='electron (measured)')
    axes[0].set_ylabel('density (n0)')
    axes[0].legend()

    axes[1].plot(bin_centers, mean_vy_i, '.', color='tab:blue', label='<vy_i> measured')
    axes[1].plot(bin_centers, mean_vy_e, '.', color='tab:red', label='<vy_e> measured')
    axes[1].axhline(v_iy, color='tab:blue', linestyle='--', alpha=0.6, label='v_iy expected')
    axes[1].axhline(v_ey, color='tab:red', linestyle='--', alpha=0.6, label='v_ey expected')
    axes[1].set_ylabel('<vy> per bin')
    axes[1].legend()

    axes[2].plot(bin_centers, Jy_expected, 'k-', label='Jy expected (from dBz/dx)', linewidth=2)
    axes[2].plot(bin_centers, Jy_measured, '.', color='tab:green', label='Jy measured (particles)')
    axes[2].set_ylabel('Jy(x)')
    axes[2].set_xlabel('x (lambda_De)')
    axes[2].legend()

    fig.suptitle('Particle loading verification against equilibrium')
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'particles_check.png'
    fig.savefig(out, dpi=150)
    print()
    print('Saved plot:', out)


if __name__ == '__main__':
    main()
