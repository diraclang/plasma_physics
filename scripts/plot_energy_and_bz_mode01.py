import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SimulationConfig, ModeSpec
from coupled_loop import run_coupled_loop


def main():
    out_dir = ROOT / 'plots'
    out_dir.mkdir(exist_ok=True)

    cfg = SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0,
        lam=10.0, B0=4.0, Bg=4.0,
        nx=64, particles_per_cell=32, dt=0.05,
        modes=[ModeSpec(m=0, n=0), ModeSpec(m=0, n=1)],
    )

    h = run_coupled_loop(
        cfg,
        n_steps=10000,
        evolve_mean=True,
        snapshot_every=500,
        log_every=500,
    )

    t = h['t']
    W_total = h['W_total_full']
    W_B = h['W_B_full']
    rel_total = (W_total - W_total[0]) / W_total[0]
    rel_B = (W_B - W_B[0]) / W_B[0]

    print('=== energy summary ===')
    print(f'W_total initial = {W_total[0]:.12e}, final = {W_total[-1]:.12e}')
    print(f'W_B_full initial = {W_B[0]:.12e}, final = {W_B[-1]:.12e}')
    print(f'rel drift total_full = {rel_total[-1]:+.6e}')
    print(f'max abs rel drift = {np.max(np.abs(rel_total)):.6e}')
    print(f'max abs rel drift B = {np.max(np.abs(rel_B)):.6e}')

    # identify the (0,1) mode in the retained mode bank
    mode_idx = None
    for i, md in enumerate(h['modes']):
        if not md.is_mean and md.m == 0 and md.n == 1:
            mode_idx = i
            break
    if mode_idx is None:
        raise RuntimeError('Could not find retained (0,1) mode in the mode bank.')

    x = np.linspace(0.0, cfg.Lx, cfg.nx, endpoint=False)
    snaps = h['snap_B']
    snap_t = h['snap_t']
    if len(snaps) == 0:
        raise RuntimeError('No snapshots were captured; snapshot_every may be too large.')

    # select a few times for the Bz mode cross-section
    selected = np.linspace(0, len(snap_t) - 1, num=5, dtype=int)
    selected = np.unique(selected)
    times = snap_t[selected]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))

    # Energy plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, W_total, lw=2, color='tab:blue', label='total energy')
    ax1.axhline(W_total[0], color='tab:blue', ls='--', lw=1, alpha=0.6)
    ax1.set_ylabel('Total energy')
    ax1.set_title('Total energy and magnetic energy vs time')
    ax1.grid(alpha=0.2)
    ax1.legend(loc='upper left')

    ax2.plot(t, W_B, lw=2, color='tab:orange', label='total magnetic energy')
    ax2.axhline(W_B[0], color='tab:orange', ls='--', lw=1, alpha=0.6)
    ax2.set_xlabel('time')
    ax2.set_ylabel('Magnetic energy')
    ax2.grid(alpha=0.2)
    ax2.legend(loc='upper left')

    fig.tight_layout()
    energy_path = out_dir / 'total_energy_and_total_magnetic_energy.png'
    fig.savefig(energy_path, dpi=200)

    # Bz(0,1) cross-sections at selected times
    fig_bz, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    for idx, color in zip(selected, colors):
        b = np.array(snaps[idx][mode_idx][2]).real
        ax.plot(x, b, lw=1.8, color=color, label=f't = {snap_t[idx]:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$B_z^{(0,1)}(x)$')
    ax.set_title(r'Cross section of $B_z(0,1)$ at selected times')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=5)
    fig_bz.tight_layout()
    bz_path = out_dir / 'bz_mode_01_cross_sections.png'
    fig_bz.savefig(bz_path, dpi=200)

    print(f'Wrote {energy_path}')
    print(f'Wrote {bz_path}')


if __name__ == '__main__':
    main()
