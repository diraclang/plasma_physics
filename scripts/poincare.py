"""Magnetic surface of section (Poincare plot) from a saved run.

Field lines of the total B at a snapshot time are traced with y as the
independent variable (needs a guide field B_y > 0):

    dx/dy = Bx/By ,   dz/dy = Bz/By

and the point (x mod Lx, z mod Lz) is recorded every time y advances by Ly
(the box period), i.e. the section is the plane y = 0 (mod Ly).
B(x,y,z) = B_mean(x) + sum_k 2 Re[ B_k(x) exp(i(ky y + kz z)) ]  (CIC-linear in x).

For fields independent of y (only m=0 modes) the flux function A_y(x,z) is also
drawn (Bx = -dAy/dz, Bz = dAy/dx): field lines lie on its contours, so this is an
independent check of the section and the only option when the guide field is 0.

    PYTHONPATH=src python scripts/poincare.py <checkpoint.npz> [n_times=3] [n_transits=60]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from coupled_loop import load_checkpoint
from equilibrium import Bz_eq


def build_field(ck, j):
    cfg, meta, s = ck['cfg'], ck['meta'], ck['snaps']
    nx = cfg.nx
    Ball = s['snap_B'][j]
    mean, pert = None, []
    for i, (m, n, is_mean) in enumerate(meta['modes']):
        if is_mean:
            mean = Ball[i].real.copy()
        else:
            pert.append((2 * np.pi * m / cfg.Ly, 2 * np.pi * n / cfg.Lz, Ball[i]))
    if mean is None:
        x = np.arange(nx) * (cfg.Lx / nx)
        mean = np.zeros((3, nx))
        mean[1] = cfg.Bg
        mean[2] = Bz_eq(x, cfg)
    return mean, pert


def field_at(mean, pert, cfg, x, y, z):
    nx = cfg.nx
    dx = cfg.Lx / nx
    sx = np.mod(x, cfg.Lx) / dx
    i0 = np.floor(sx).astype(int) % nx
    i1 = (i0 + 1) % nx
    f = sx - np.floor(sx)
    B = (1 - f) * mean[:, i0] + f * mean[:, i1]
    for ky, kz, F in pert:
        Fi = (1 - f) * F[:, i0] + f * F[:, i1]
        B = B + 2.0 * np.real(Fi * np.exp(1j * (ky * y + kz * z)))
    return B


def poincare(mean, pert, cfg, x0, z0, n_transits, dy=4.0):
    steps = int(round(cfg.Ly / dy))
    h = cfg.Ly / steps
    x, z = x0.copy(), z0.copy()
    pts = np.zeros((n_transits + 1, 2, len(x0)))
    pts[0] = x, z
    y = 0.0

    def rhs(y, x, z):
        B = field_at(mean, pert, cfg, x, y, z)
        return B[0] / B[1], B[2] / B[1]

    for k in range(n_transits):
        for _ in range(steps):
            k1x, k1z = rhs(y, x, z)
            k2x, k2z = rhs(y + 0.5 * h, x + 0.5 * h * k1x, z + 0.5 * h * k1z)
            k3x, k3z = rhs(y + 0.5 * h, x + 0.5 * h * k2x, z + 0.5 * h * k2z)
            k4x, k4z = rhs(y + h, x + h * k3x, z + h * k3z)
            x = x + h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            z = z + h * (k1z + 2 * k2z + 2 * k3z + k4z) / 6
            y += h
        pts[k + 1] = np.mod(x, cfg.Lx), np.mod(z, cfg.Lz)
    return pts


def flux_function(mean, pert, cfg, nz=257):
    nx = cfg.nx
    dx = cfg.Lx / nx
    Bz0 = mean[2]
    A0 = np.concatenate(([0.0], np.cumsum(0.5 * (Bz0[1:] + Bz0[:-1]) * dx)))
    A0 = A0 - A0.mean()
    zg = np.linspace(0, cfg.Lz, nz)
    A = np.repeat(A0[:, None], nz, axis=1)
    for ky, kz, F in pert:
        if abs(ky) > 1e-12 or abs(kz) < 1e-12:
            return None
        A = A + 2.0 * np.real((1j * F[0] / kz)[:, None] * np.exp(1j * kz * zg)[None, :])
    return zg, A


def main():
    path = Path(sys.argv[1])
    n_times = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_transits = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    ck = load_checkpoint(path)
    cfg, s = ck['cfg'], ck['snaps']
    ns = len(s['snap_t'])
    pick = np.unique(np.linspace(0, ns - 1, n_times).astype(int))
    x = np.arange(cfg.nx) * (cfg.Lx / cfg.nx)
    have_guide = cfg.Bg > 0.5

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrow = 2 if have_guide else 1
    fig, axs = plt.subplots(nrow, len(pick), figsize=(5.6 * len(pick), 5.2 * nrow), squeeze=False)

    # start points: lines across x at three z values (dense around both sheets)
    xs = np.unique(np.r_[np.linspace(0, cfg.Lx, 40, endpoint=False),
                         np.linspace(cfg.Lx / 4 - 45, cfg.Lx / 4 + 45, 45),
                         np.linspace(3 * cfg.Lx / 4 - 45, 3 * cfg.Lx / 4 + 45, 45)])
    zs = [0.0, cfg.Lz / 3, 2 * cfg.Lz / 3]
    x0 = np.tile(xs, len(zs))
    z0 = np.repeat(zs, len(xs))

    for c, j in enumerate(pick):
        mean, pert = build_field(ck, j)
        r = 0
        if have_guide:
            pts = poincare(mean, pert, cfg, x0, z0, n_transits)
            ax = axs[0, c]
            ax.scatter(pts[:, 0].ravel(), pts[:, 1].ravel(), s=0.5, c=np.tile(x0, n_transits + 1), cmap='turbo')
            ax.set_title(f'Poincare section y=0, t={s["snap_t"][j]:.0f} ({n_transits} transits)')
            ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_xlim(0, cfg.Lx); ax.set_ylim(0, cfg.Lz)
            r = 1
        ff = flux_function(mean, pert, cfg)
        ax = axs[r, c]
        if ff is not None:
            zg, A = ff
            ax.contour(x, zg, A.T, levels=60, colors='k', linewidths=0.5)
            ax.set_title(f'flux contours A_y(x,z), t={s["snap_t"][j]:.0f}')
        else:
            ax.set_title('flux function n/a (modes with ky != 0)')
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_xlim(0, cfg.Lx)
    fig.tight_layout()
    out = path.parent / 'poincare.png'
    fig.savefig(out, dpi=110)
    print(f'wrote {out}  (snapshot times {[round(float(s["snap_t"][j])) for j in pick]})')


if __name__ == '__main__':
    main()
