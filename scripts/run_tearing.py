"""Tearing-mode run on the symmetric double Harris sheet.

20,000 pushed particles per species (the mirror sheet is deposited only as an
image), 1993 geometry, dynamic (0,0) mean field, quiet start, a seed well above
the shot-noise floor and a small mode bank.  Reports energy partition, heating
and the growth of B_x at the sheet centres; saves tearing_run.npz / .png.

    PYTHONPATH=src python scripts/run_tearing.py [n_steps] [ppc] [seed_amp]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from config import SimulationConfig, mode_range
from coupled_loop import run_coupled_loop


def smooth(a, n=100):
    k = np.ones(n)
    return np.convolve(a, k, mode='same') / np.convolve(np.ones_like(a), k, mode='same')


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    ppc = int(sys.argv[2]) if len(sys.argv) > 2 else 306        # ~20,000 pushed per species
    seed_amp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05  # Bx ~ 4*seed*n
    modeset = sys.argv[4] if len(sys.argv) > 4 else 'small'       # 'small' (m,n in -1..1), 'full' (m -3..3), 'tearing' (m=0, n=+-1)
    Bg = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0          # guide field By
    mrange = {'full': range(-3, 4), 'tearing': range(0, 1)}.get(modeset, range(-1, 2))
    tag = f'tearing_{modeset}_Bg{Bg:g}'
    lam = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0         # sheet scale length
    Lx = float(sys.argv[7]) if len(sys.argv) > 7 else 256.0         # box (dx = 1 => nx = Lx)
    Lz = float(sys.argv[8]) if len(sys.argv) > 8 else 256.0         # Ly = Lz
    if lam != 20.0 or Lx != 256.0 or Lz != 256.0:
        tag += f'_lam{lam:g}_Lx{Lx:g}_Lz{Lz:g}'
    tag += f'_ppc{ppc}'
    if seed_amp != 0.05:
        tag += f'_seed{seed_amp:g}'
    static = len(sys.argv) > 10 and sys.argv[10] == 'static'      # frozen analytic B_eq, no dynamic (0,0) mean field
    if static:
        tag += '_static'
    n_bg = float(sys.argv[11]) if len(sys.argv) > 11 else 0.2       # uniform background density (units of n0)
    if n_bg != 0.2:
        tag += f'_nbg{n_bg:g}'
    resume = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9] != 'none' else None   # checkpoint to continue from (n_steps = TOTAL)
    out_dir = ROOT / 'runs' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / tag)

    cfg = SimulationConfig(
        Lx=Lx, Ly=Lz, Lz=Lz, lam=lam, B0=4.0, Bg=Bg, n_bg=n_bg,
        nx=int(Lx), particles_per_cell=ppc, dt=0.05,
        modes=mode_range(mrange, range(-1, 2), seed_amplitude=seed_amp),
    )
    h = run_coupled_loop(cfg, n_steps=n_steps, evolve_mean=not static, paired_load=True,
                         symmetric=True, log_every=max(1, n_steps // 25),
                         snapshot_every=max(1, n_steps // 50),
                         checkpoint_prefix=prefix, checkpoint_every=2000, keep_every=10000,
                         resume_from=resume)

    t, W = h['t'], h['W_total']
    rel = (W - W[0]) / W[0]
    labels = [f'({md.m},{md.n})' for md in h['modes'] if not md.is_mean]

    print('\n=== tearing run summary ===')
    print(f'steps={n_steps} dt={h["dt"]:.4f} t_end={t[-1]:.1f}  pushed/species={h["species"][0].n}  '
          f'Bg={Bg:g} modes={modeset} lam={lam:g} Lx={Lx:g} Lz={Lz:g} k*lam={2 * np.pi / Lz * lam:.2f}  '
          f'seed Bx~{seed_amp * cfg.B0 * cfg.Lx / cfg.Lz:.3f}  symmetry error={h["sym_err"]:.1e}')
    print(f'total energy drift: final {rel[-1]:+.3e}  max {np.max(np.abs(rel)):.3e}')
    for name in ('W_K', 'W_B0', 'W_E', 'W_B'):
        print(f'{name:5s}: {h[name][0]:.5e} -> {h[name][-1]:.5e}  '
              f'({(h[name][-1] - h[name][0]) / W[0]:+.3e} of W_total)')
    for name in ('T_ion', 'T_ele'):
        print(f'{name}: {h[name][0]:.4f} -> {h[name][-1]:.4f}')
    print('mode      |Bx| at x=Lx/4: start -> end (x growth) | max over t')
    for k, lab in enumerate(labels):
        s = smooth(h['bx_s1'][:, k])
        print(f'  {lab:8s} {np.mean(h["bx_s1"][:50, k]):.3e} -> {np.mean(h["bx_s1"][-50:, k]):.3e} '
              f'(x{np.mean(h["bx_s1"][-50:, k]) / np.mean(h["bx_s1"][:50, k]):.2f}) | max {s.max():.3e}')
        es = smooth(h['mode_E'][:, k] + h['mode_B'][:, k])
        for f0, f1 in ((0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.97)):
            sl = slice(int(f0 * n_steps), int(f1 * n_steps))
            p = np.polyfit(t[sl], np.log(es[sl]), 1)[0] / 2.0     # amplitude-equivalent rate
            print(f'            gamma(amp) over t={t[sl][0]:6.0f}..{t[sl][-1]:6.0f}: {p:+.3e}   '
                  f'E+B {es[sl][0]:.3e} -> {es[sl][-1]:.3e}')

    np.savez(out_dir / f'{tag}.npz', **{k: h[k] for k in
             ('t', 'W_K', 'W_E', 'W_B', 'W_B0', 'W_total', 'T_ion', 'T_ele', 'amp', 'bx_s1',
              'bx_s2', 'mode_E', 'mode_B', 'snap_t', 'snap_Bx')})
    np.save(out_dir / f'{tag}_labels.npy', np.array(labels))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, rel); ax[0, 0].set_title('total energy drift')
    ax[0, 1].plot(t, h['W_B0'] - h['W_B0'][0], label='mean magnetic dW_B0')
    ax[0, 1].plot(t, h['W_K'] - h['W_K'][0], label='kinetic dW_K')
    ax[0, 1].plot(t, h['W_E'] + h['W_B'] - h['W_E'][0] - h['W_B'][0], label='pert. fields')
    ax[0, 1].legend(); ax[0, 1].set_title('energy exchange')
    ax[1, 0].plot(t, h['T_ion'], label='ion'); ax[1, 0].plot(t, h['T_ele'], label='electron')
    ax[1, 0].legend(); ax[1, 0].set_title('thermal energy per particle')
    for k, lab in enumerate(labels):
        ax[1, 1].semilogy(t, smooth(h['bx_s1'][:, k]) + 1e-30, label=lab)
    ax[1, 1].legend(); ax[1, 1].set_title('|Bx_k| at sheet centre x=Lx/4 (100-step mean)')
    for a in ax.ravel():
        a.set_xlabel('t')
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}.png', dpi=110)

    # per-mode energy histories (electric + magnetic, 100-step mean)
    fig2, bx = plt.subplots(1, 2, figsize=(14, 5))
    for k, lab in enumerate(labels):
        e = smooth(h['mode_E'][:, k] + h['mode_B'][:, k])
        bx[0].semilogy(t, e + 1e-30, label=lab)
        bx[1].semilogy(t, smooth(h['mode_B'][:, k]) + 1e-30, label=lab)
    bx[0].set_title('energy per mode (E + B)'); bx[1].set_title('magnetic energy per mode')
    for a in bx:
        a.set_xlabel('t'); a.legend(fontsize=7, ncol=2)
    fig2.tight_layout(); fig2.savefig(out_dir / f'{tag}_modes.png', dpi=110)

    # Bx_k(x) profile of the (0,+-1) mode at the snapshot times
    from equilibrium import Bz_eq
    xg = np.arange(cfg.nx) * (cfg.Lx / cfg.nx)
    k0 = next((i for i, lab in enumerate(labels) if lab.startswith('(0,')), 0)
    fig3, bxax = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    cm = plt.cm.viridis(np.linspace(0, 1, len(h['snap_t'])))
    for s, ts in enumerate(h['snap_t']):
        b = h['snap_Bx'][s, k0]            # Bx_(0,-1)(x); Bx_(0,+1) = conj of it
        bxax[0].plot(xg, np.conj(b).real, color=cm[s], label=f't={ts:.0f}')
        bxax[1].plot(xg, np.conj(b).imag, color=cm[s])
        bxax[2].plot(xg, np.abs(b), color=cm[s])
    bxax[0].set_title(f'Re Bx_(0,+1)(x)   [mode {labels[k0]} conjugated]')
    bxax[1].set_title('Im Bx_(0,+1)(x)'); bxax[2].set_title('|Bx_(0,1)(x)|')
    for a in bxax:
        a2 = a.twinx(); a2.plot(xg, Bz_eq(xg, cfg), 'k--', lw=0.8, alpha=0.5); a2.set_ylabel('Bz_eq(x)')
        for xr in cfg.resonant_x(h['modes'][k0]):
            a.axvline(xr, color='r', ls=':', lw=0.8)
        a.set_xlabel('x'); a.set_ylabel('Bx')
    bxax[0].legend(fontsize=7)
    fig3.tight_layout(); fig3.savefig(out_dir / f'{tag}_bx_profile.png', dpi=110)
    print(f'saved to {out_dir}: {tag}.npz, {tag}.png, {tag}_modes.png, {tag}_bx_profile.png, '
          f'{tag}_final.npz (restartable) and checkpoints')


if __name__ == '__main__':
    main()
