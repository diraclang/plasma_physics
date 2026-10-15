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
        Lx=128.0, Ly=128.0, Lz=128.0,
        lam=10.0, B0=4.0, Bg=4.0,
        nx=128, particles_per_cell=32, dt=0.05,
        modes=[ModeSpec(m=0, n=0), ModeSpec(m=0, n=1), ModeSpec(m=0, n=-1)],
    )

    h = run_coupled_loop(
        cfg,
        n_steps=10000,
        evolve_mean=True,
        snapshot_every=500,
        log_every=1000,
    )

    x = np.linspace(0.0, cfg.Lx, cfg.nx, endpoint=False)
    dx = cfg.Lx / cfg.nx
    area = dx * cfg.Ly * cfg.Lz

    # build density profiles from the particle moments at each saved snapshot
    profile_times = []
    density_profiles = []
    for snap_t, prof in zip(h['snap_t'], h['snap_prof']):
        dens = np.zeros(cfg.nx)
        for sp, moments in zip(h['species'], prof):
            if sp.species != 'ion':
                continue
            count = moments[0]
            dens += count * sp.weight / area
        profile_times.append(snap_t)
        density_profiles.append(dens)

    density_profiles = np.array(density_profiles)
    selected = np.unique(np.linspace(0, len(profile_times) - 1, num=6, dtype=int))
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))

    out_dir = ROOT / 'plots'
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, col in zip(selected, colors):
        ax.plot(x, density_profiles[idx], lw=2, color=col, label=f't={profile_times[idx]:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('n(x)')
    ax.set_title('Density profile evolution for the large-box mean + (0,1),(0,-1) run')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path = out_dir / 'density_evolution_large_box.png'
    fig.savefig(path, dpi=220)
    print(f'Wrote {path}')
    print(f'num snapshots = {len(profile_times)}')
    print(f'initial density max = {density_profiles[0].max():.6e}, final density max = {density_profiles[-1].max():.6e}')
    print(f'initial density min = {density_profiles[0].min():.6e}, final density min = {density_profiles[-1].min():.6e}')


if __name__ == '__main__':
    main()
