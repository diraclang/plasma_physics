"""Long coupled run with the dynamic (0,0) mean field.

Reports energy partition, species heating and mode growth, saves
long_run.npz and long_run.png in the project root.

    PYTHONPATH=src python scripts/run_long.py [n_steps] [ppc]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from config import SimulationConfig, mode_range
from coupled_loop import run_coupled_loop


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    ppc = int(sys.argv[2]) if len(sys.argv) > 2 else 153   # ~20,000 particles/species at nx=256
    resume = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'none' else None

    # 1993 geometry: Lx=Ly=Lz=256, lambda=20, dx=lambda_De; (0,1) resonates at x=64,192
    cfg = SimulationConfig(
        Lx=256.0, Ly=256.0, Lz=256.0, lam=20.0, B0=4.0, Bg=4.0,
        nx=256, particles_per_cell=ppc, dt=0.1,
        modes=mode_range(range(-3, 4), range(-1, 2), seed_amplitude=1.0e-3),
    )
    run_dir = ROOT / 'runs' / f'long_run_ppc{ppc}'
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f'long_run_ppc{ppc}')
    h = run_coupled_loop(cfg, n_steps=n_steps, evolve_mean=True, paired_load=True,
                         log_every=max(1, n_steps // 25),
                         checkpoint_prefix=prefix, checkpoint_every=1000, keep_every=5000,
                         resume_from=resume)

    t = h['t']
    W = h['W_total']
    rel = (W - W[0]) / W[0]
    pert_labels = [f'({md.m},{md.n})' for md in h['modes'] if not md.is_mean]

    print('\n=== long run summary ===')
    print(f'steps={n_steps}  dt={h["dt"]:.4f}  t_end={t[-1]:.1f}  ppc={ppc}  '
          f'particles/species={h["species"][0].n}')
    print(f'total energy drift: final {rel[-1]:+.3e}  max {np.max(np.abs(rel)):.3e}')
    for name in ('W_K', 'W_B0', 'W_E', 'W_B'):
        print(f'{name:5s}: {h[name][0]:.5e} -> {h[name][-1]:.5e}  '
              f'({(h[name][-1] - h[name][0]) / W[0]:+.3e} of W_total)')
    for name in ('T_ion', 'T_ele'):
        print(f'{name}: {h[name][0]:.4f} -> {h[name][-1]:.4f}  '
              f'(x{h[name][-1] / h[name][0]:.3f})')
    amp = h['amp']
    print('mode  max|Bx_k|: start -> end  (growth factor)')
    for k, lab in enumerate(pert_labels):
        a0, a1 = np.mean(amp[:50, k]), np.mean(amp[-50:, k])
        print(f'  {lab:8s} {a0:.3e} -> {a1:.3e}  (x{a1 / a0:.2f})')

    np.savez(ROOT / 'long_run.npz', **{k: h[k] for k in
             ('t', 'W_K', 'W_E', 'W_B', 'W_B0', 'W_total', 'K_ion', 'K_ele',
              'T_ion', 'T_ele', 'amp')})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, rel)
    ax[0, 0].set_title('total energy drift (W - W0)/W0')
    ax[0, 1].plot(t, h['W_B0'] - h['W_B0'][0], label='mean-field magnetic dW_B0')
    ax[0, 1].plot(t, h['W_K'] - h['W_K'][0], label='kinetic dW_K')
    ax[0, 1].plot(t, h['W_E'] + h['W_B'] - h['W_E'][0] - h['W_B'][0], label='pert. field d(W_E+W_B)')
    ax[0, 1].legend(); ax[0, 1].set_title('energy exchange')
    ax[1, 0].plot(t, h['T_ion'], label='ion'); ax[1, 0].plot(t, h['T_ele'], label='electron')
    ax[1, 0].legend(); ax[1, 0].set_title('thermal energy per particle (flow removed)')
    for k, lab in enumerate(pert_labels):
        ax[1, 1].semilogy(t, amp[:, k] + 1e-30, label=lab)
    ax[1, 1].legend(fontsize=6, ncol=2); ax[1, 1].set_title('max |Bx_k| per mode')
    for a in ax.ravel():
        a.set_xlabel('t')
    fig.tight_layout()
    fig.savefig(ROOT / 'long_run.png', dpi=110)
    print('saved long_run.npz, long_run.png')


if __name__ == '__main__':
    main()
