"""Analyse a saved run from its checkpoint (<prefix>_ckpt_latest.npz,
<prefix>_ckpt_<step>.npz or <prefix>_final.npz) -- no re-run needed, and it
works on a run that is still in progress.

    PYTHONPATH=src python scripts/analyze_run.py runs/<tag>/<tag>_ckpt_latest.npz

Writes analysis_*.png next to the checkpoint and prints the key numbers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from coupled_loop import load_checkpoint
from equilibrium import Bz_eq


def smooth(a, n=200):
    k = np.ones(min(n, max(1, len(a))))
    return np.convolve(a, k, mode='same') / np.convolve(np.ones_like(a), k, mode='same')


def main():
    path = Path(sys.argv[1])
    ck = load_checkpoint(path)
    cfg, meta, h, s = ck['cfg'], ck['meta'], ck['hist'], ck['snaps']
    out = path.parent
    t = h['t']
    n = len(t)
    nx = cfg.nx
    x = np.arange(nx) * (cfg.Lx / nx)
    dx = cfg.Lx / nx
    labels = [f'({m},{nn})' for m, nn, mean in meta['modes'] if not mean]
    mean_idx = next((i for i, md in enumerate(meta['modes']) if md[2]), None)
    pert_idx = [i for i, md in enumerate(meta['modes']) if not md[2]]
    W = h['W_total']
    rel = (W - W[0]) / W[0]
    sym = meta['flags']['symmetric']

    print(f'=== {path.name}: step {meta["step"]}  t={t[-1]:.1f}  dt={meta["dt"]:.4f}  '
          f'Lx={cfg.Lx:g} Lz={cfg.Lz:g} lam={cfg.lam:g} Bg={cfg.Bg:g}  ppc={cfg.particles_per_cell}  '
          f'pushed/species={len(ck["species"][0]["x"])}  symmetric={sym}')
    print(f'total energy drift: final {rel[-1]:+.3e}  max {np.max(np.abs(rel)):.3e}')
    for name in ('W_K', 'W_B0', 'W_E', 'W_B'):
        print(f'  {name:5s}: {h[name][0]:.5e} -> {h[name][-1]:.5e}  ({(h[name][-1] - h[name][0]) / W[0]:+.3e} of W_total)')
    print(f'  T_ion {h["T_ion"][0]:.4f} -> {h["T_ion"][-1]:.4f}   T_ele {h["T_ele"][0]:.4f} -> {h["T_ele"][-1]:.4f}')

    print('mode energy growth (E+B), amplitude-equivalent rate gamma = 0.5 dln(E+B)/dt:')
    es = {lab: smooth(h['mode_E'][:, k] + h['mode_B'][:, k]) for k, lab in enumerate(labels)}
    for lab in labels:
        row = []
        for f0, f1 in ((0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.97)):
            i0, i1 = int(f0 * n), int(f1 * n)
            if i1 - i0 > 10:
                row.append(f'{0.5 * np.polyfit(t[i0:i1], np.log(es[lab][i0:i1]), 1)[0]:+.2e}')
        print(f'  {lab:8s} E+B {es[lab][int(0.05 * n)]:.3e} -> {es[lab][-1]:.3e}   gamma by quarter: {" ".join(row)}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. energies / heating
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, rel); ax[0, 0].set_title('total energy drift (W-W0)/W0')
    ax[0, 1].plot(t, h['W_B0'] - h['W_B0'][0], label='mean magnetic')
    ax[0, 1].plot(t, h['W_K'] - h['W_K'][0], label='kinetic')
    ax[0, 1].plot(t, h['W_E'] + h['W_B'] - h['W_E'][0] - h['W_B'][0], label='perturbation fields')
    ax[0, 1].legend(); ax[0, 1].set_title('energy exchange')
    ax[1, 0].plot(t, h['T_ion'], label='ion'); ax[1, 0].plot(t, h['T_ele'], label='electron')
    ax[1, 0].legend(); ax[1, 0].set_title('thermal energy per particle')
    for lab in labels:
        ax[1, 1].semilogy(t, es[lab], label=lab)
    ax[1, 1].legend(fontsize=7, ncol=2); ax[1, 1].set_title('energy per mode (E+B)')
    for a in ax.ravel():
        a.set_xlabel('t')
    fig.tight_layout(); fig.savefig(out / 'analysis_energy.png', dpi=110)

    ns = len(s['snap_t'])
    if ns:
        cm = plt.cm.viridis(np.linspace(0, 1, ns))
        # 2. Bx(x) of the first m=0 mode: profile and change from the first snapshot
        k0 = next((i for i, l in enumerate(labels) if l.startswith('(0,')), 0)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
        b0 = np.conj(s['snap_Bx'][0, k0])
        for j in range(ns):
            b = np.conj(s['snap_Bx'][j, k0])
            ax[0].plot(x, np.abs(b), color=cm[j], label=f't={s["snap_t"][j]:.0f}')
            ax[1].plot(x, (b - b0).real, color=cm[j])
            ax[2].plot(x, (b - b0).imag, color=cm[j])
        ax[0].set_title('|Bx_(0,1)(x)|'); ax[1].set_title('Re[Bx(t)-Bx(0)]'); ax[2].set_title('Im[Bx(t)-Bx(0)]')
        for xr in cfg.resonant_x(type('M', (), {'m': 0, 'n': 1})()):
            for a in ax:
                a.axvline(xr, color='r', ls=':', lw=0.8)
        ax[0].legend(fontsize=6, ncol=2)
        for a in ax:
            a.set_xlabel('x')
        fig.tight_layout(); fig.savefig(out / 'analysis_bx_profile.png', dpi=110)

        # 3. mean-field profiles and the current they imply (Ampere) vs the particle current
        if mean_idx is not None:
            fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
            for j in (0, ns // 2, ns - 1):
                Bm = s['snap_B'][j, mean_idx]
                ax[0].plot(x, Bm[2].real, label=f'Bz t={s["snap_t"][j]:.0f}')
                jy_a = -(np.roll(Bm[2].real, -1) - np.roll(Bm[2].real, 1)) / (2 * dx)
                prof = s['snap_prof'][j]
                jy_p = sum(sp['q'] * sp['weight'] * prof[i, 2] for i, sp in enumerate(meta['species'])) / (dx * cfg.Ly * cfg.Lz)
                ax[1].plot(x, jy_a, label=f'-dBz/dx t={s["snap_t"][j]:.0f}')
                ax[1].plot(x, jy_p, '--', alpha=0.6, label=f'particle Jy t={s["snap_t"][j]:.0f}')
                dens = prof[0, 0] * meta['species'][0]['weight'] / (dx * cfg.Ly * cfg.Lz)
                ax[2].plot(x, dens, label=f'ion n(x) t={s["snap_t"][j]:.0f}')
            ax[0].set_title('mean Bz(x)'); ax[1].set_title('current: Ampere vs particles'); ax[2].set_title('ion density')
            for a in ax:
                a.legend(fontsize=6); a.set_xlabel('x')
            fig.tight_layout(); fig.savefig(out / 'analysis_mean_profiles.png', dpi=110)
            dB = s['snap_B'][-1, mean_idx][2].real - s['snap_B'][0, mean_idx][2].real
            print(f'mean Bz(x) change first->last snapshot: max |dBz| = {np.max(np.abs(dB)):.3e}  (B0={cfg.B0:g})')
    print(f'wrote analysis_energy.png, analysis_bx_profile.png, analysis_mean_profiles.png to {out}')


if __name__ == '__main__':
    main()
