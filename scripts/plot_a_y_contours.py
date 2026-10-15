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
    cfg = SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0,
        lam=10.0, B0=4.0, Bg=4.0,
        nx=64, particles_per_cell=32, dt=0.05,
        modes=[ModeSpec(m=0, n=0), ModeSpec(m=0, n=1, seed_amplitude=0.0)],
    )

    h = run_coupled_loop(
        cfg,
        n_steps=10000,
        evolve_mean=True,
        snapshot_every=1000,
        log_every=500,
    )

    mode_idx = None
    for i, md in enumerate(h['modes']):
        if not md.is_mean and md.m == 0 and md.n == 1:
            mode_idx = i
            break
    if mode_idx is None:
        raise RuntimeError('No retained (0,1) mode found.')

    kz = cfg.k_z(h['modes'][mode_idx])
    x = np.linspace(0.0, cfg.Lx, cfg.nx, endpoint=False)
    z = np.linspace(0.0, cfg.Lz, 256, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    times = [0, len(h['snap_t']) // 2, len(h['snap_t']) - 1]
    times = [int(t) for t in times if t < len(h['snap_t'])]
    times = sorted(set(times))

    out_dir = ROOT / 'plots'
    out_dir.mkdir(exist_ok=True)

    for idx in times:
        B = h['snap_B'][idx][mode_idx]
        Ay_mode = 1j * B[0] / kz     # from Bx = -i kz Ay
        # 2D real flux function A_y(x,z) = 2 Re[Ay_mode(x) e^{i kz z}]
        phase = np.exp(1j * kz * z[None, :])
        Ay2d = 2.0 * np.real(Ay_mode[:, None] * phase)

        fig, ax = plt.subplots(figsize=(8, 6))
        levels = np.linspace(np.min(Ay2d), np.max(Ay2d), 40)
        cf = ax.contourf(X, Z, Ay2d, levels=levels, cmap='RdBu_r', alpha=0.9)
        cs = ax.contour(X, Z, Ay2d, levels=levels[::4], colors='k', linewidths=0.6, alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(r'Contour plot of $A_y(x,z)$ at $t = %.1f$' % h['snap_t'][idx])
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r'$A_y$')
        fig.tight_layout()
        path = out_dir / f'a_y_contour_t_{idx:03d}.png'
        fig.savefig(path, dpi=200)
        print(f'Wrote {path}')

    # One combined figure for comparison
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4), sharex=True, sharey=True)
    if len(times) == 1:
        axes = [axes]
    for ax, idx in zip(axes, times):
        B = h['snap_B'][idx][mode_idx]
        Ay_mode = 1j * B[0] / kz
        phase = np.exp(1j * kz * z[None, :])
        Ay2d = 2.0 * np.real(Ay_mode[:, None] * phase)
        levels = np.linspace(np.min(Ay2d), np.max(Ay2d), 30)
        cf = ax.contourf(X, Z, Ay2d, levels=levels, cmap='RdBu_r')
        ax.contour(X, Z, Ay2d, levels=levels[::5], colors='k', linewidths=0.6, alpha=0.7)
        ax.set_title(f't = {h["snap_t"][idx]:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    fig.suptitle(r'Contours of $A_y(x,z)$ for the $(0,1)$ mode')
    fig.colorbar(cf, ax=axes, shrink=0.9)
    fig.tight_layout()
    path_all = out_dir / 'a_y_contours_all_times.png'
    fig.savefig(path_all, dpi=200)
    print(f'Wrote {path_all}')


if __name__ == '__main__':
    main()
