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


def reconstruct_Ay_mode_from_B(mode_B: np.ndarray, kz: float):
    """Return A_y(x) for one retained Fourier mode using Bx = -i kz Ay."""
    return 1j * mode_B[0] / kz


def main():
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

    mode_idx = None
    for i, md in enumerate(h['modes']):
        if not md.is_mean and md.m == 0 and md.n == 1:
            mode_idx = i
            break
    if mode_idx is None:
        raise RuntimeError('No retained (0,1) mode found.')

    kz = cfg.k_z(h['modes'][mode_idx])
    x = np.linspace(0.0, cfg.Lx, cfg.nx, endpoint=False)
    z = np.linspace(0.0, cfg.Lz, 128, endpoint=False)
    Z, X = np.meshgrid(z, x, indexing='xy')

    # choose a few representative snapshots at increasing time
    idxs = np.unique(np.linspace(0, len(h['snap_t']) - 1, num=4, dtype=int))
    out_dir = ROOT / 'plots'
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    for ax, j in zip(axes.ravel(), idxs):
        B = h['snap_B'][j][mode_idx]
        Ay_mode = reconstruct_Ay_mode_from_B(B, kz)
        A_y = 2.0 * np.real(Ay_mode[:, None] * np.exp(1j * kz * Z[:, :]))
        surf = ax.plot_surface(X, Z, A_y, cmap='viridis', linewidth=0, antialiased=True)
        ax.set_title(f't = {h["snap_t"][j]:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel(r'$A_y$')
        fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.08)

    fig.suptitle(r'Reconstructed $A_y(x,z)$ for the retained $(0,1)$ mode', fontsize=13)
    fig.tight_layout()
    path = out_dir / 'a_y_surface_mode_01.png'
    fig.savefig(path, dpi=200)

    # Also save the x-profile at selected times for easier comparison
    fig2, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(idxs)))
    for j, col in zip(idxs, colors):
        B = h['snap_B'][j][mode_idx]
        Ay_mode = reconstruct_Ay_mode_from_B(B, kz)
        ayx = 2.0 * np.real(Ay_mode)  # x-only profile at z=0
        ax.plot(x, ayx, lw=2, color=col, label=f't={h["snap_t"][j]:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$A_y(x,z=0)$')
    ax.set_title(r'$A_y$ cross-section at z=0 for the $(0,1)$ mode')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    path2 = out_dir / 'a_y_cross_section_mode_01.png'
    fig2.savefig(path2, dpi=200)

    print(f'Wrote {path}')
    print(f'Wrote {path2}')


if __name__ == '__main__':
    main()
