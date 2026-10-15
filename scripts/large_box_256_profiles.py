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
        Lx=256.0, Ly=256.0, Lz=256.0,
        lam=20.0, B0=4.0, Bg=4.0,
        nx=256, particles_per_cell=153, dt=0.05,
        modes=[ModeSpec(m=0, n=0), ModeSpec(m=0, n=1), ModeSpec(m=0, n=-1)],
    )

    run_dir = ROOT / 'runs' / 'large_box_256_profiles'
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / 'large_box_256_profiles')
    h = run_coupled_loop(
        cfg,
        n_steps=5000,
        evolve_mean=True,
        snapshot_every=500,
        log_every=500,
        checkpoint_prefix=prefix,
        checkpoint_every=1000,
        keep_every=2500,
    )

    x = np.linspace(0.0, cfg.Lx, cfg.nx, endpoint=False)
    dx = cfg.Lx / cfg.nx
    area = dx * cfg.Ly * cfg.Lz
    prof_final = h['snap_prof'][-1]
    n_total = np.zeros(cfg.nx)
    Jy_total = np.zeros(cfg.nx)
    for sp, moments in zip(h['species'], prof_final):
        n_total += moments[0] * sp.weight / area
        Jy_total += sp.q * sp.weight * moments[2] / area

    mode_idx = None
    for i, md in enumerate(h['modes']):
        if not md.is_mean and md.m == 0 and md.n == 1:
            mode_idx = i
            break
    if mode_idx is None:
        raise RuntimeError('No (0,1) mode found.')

    Bx_final = np.real(h['snap_Bx'][-1, mode_idx])

    out_dir = ROOT / 'plots'
    out_dir.mkdir(exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax1.plot(x, Bx_final, lw=2, color='tab:blue')
    ax1.set_ylabel(r'$B_x^{(0,1)}(x)$')
    ax1.set_title(r'Final cross sections at $t = %.1f$' % h['snap_t'][-1])
    ax1.grid(alpha=0.2)

    ax2.plot(x, n_total, lw=2, color='tab:green')
    ax2.set_ylabel(r'$n(x)$')
    ax2.grid(alpha=0.2)

    ax3.plot(x, Jy_total, lw=2, color='tab:orange')
    ax3.set_xlabel('x')
    ax3.set_ylabel(r'$J_y(x)$')
    ax3.grid(alpha=0.2)
    fig.tight_layout()
    path = out_dir / 'large_box_256_final_profiles.png'
    fig.savefig(path, dpi=220)
    print(f'Wrote {path}')
    print(f'final time = {h["snap_t"][-1]:.1f}')
    print(f'B_x max = {np.max(np.abs(Bx_final)):.6e}')
    print(f'n range = {n_total.min():.6e} .. {n_total.max():.6e}')
    print(f'J_y range = {Jy_total.min():.6e} .. {Jy_total.max():.6e}')


if __name__ == '__main__':
    main()
