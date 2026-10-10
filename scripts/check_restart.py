"""Restart test: a run interrupted at a checkpoint and resumed must reproduce an
uninterrupted run exactly (the loop uses no random numbers after loading).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

import coupled_loop as cl
from config import SimulationConfig, mode_range


def make_cfg():
    return SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0, lam=10.0, B0=4.0, Bg=4.0, nx=64,
        particles_per_cell=50, dt=0.05,
        modes=mode_range(range(-1, 2), range(-1, 2), seed_amplitude=0.05),
    )


def main():
    cfg = make_cfg()
    kw = dict(evolve_mean=True, paired_load=True, symmetric=True, snapshot_every=20)
    full = cl.run_coupled_loop(cfg, n_steps=60, **kw)

    with tempfile.TemporaryDirectory() as d:
        pre = str(Path(d) / 'run')
        cl.run_coupled_loop(cfg, n_steps=30, checkpoint_prefix=pre, checkpoint_every=30, **kw)
        resumed = cl.run_coupled_loop(cfg, n_steps=60, resume_from=f'{pre}_ckpt_latest.npz',
                                      checkpoint_prefix=pre, checkpoint_every=30, **kw)
        ck = cl.load_checkpoint(f'{pre}_final.npz')

    print('=== checkpoint / restart ===')
    ok = True
    for k in ('t', 'W_K', 'W_E', 'W_B', 'W_B0', 'W_total', 'T_ion', 'T_ele', 'amp', 'mode_E', 'mode_B'):
        diff = float(np.max(np.abs(full[k] - resumed[k])))
        print(f'  {k:8s} max |straight - resumed| = {diff:.1e}')
        ok &= diff == 0.0
    for a in ('x', 'vy'):
        diff = float(np.max(np.abs(getattr(full['species'][0], a) - getattr(resumed['species'][0], a))))
        print(f'  particle {a:3s} max diff = {diff:.1e}')
        ok &= diff == 0.0
    print(f'  final checkpoint holds {len(ck["hist"]["t"])} steps of history, '
          f'{len(ck["snaps"]["snap_t"])} snapshots, snap_prof shape {ck["snaps"]["snap_prof"].shape}')
    ok &= len(ck['hist']['t']) == 60
    if not ok:
        raise SystemExit('restart does not reproduce the uninterrupted run')
    print('PASS')


if __name__ == '__main__':
    main()
