from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot time history of equatorial particle motion.")
    p.add_argument("--input", type=str, default="hybrid/output/ensemble.npz")
    p.add_argument("--z-band", type=float, default=0.2, help="Equatorial band half-thickness: |z| <= z-band")
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--bins-r", type=int, default=140)
    p.add_argument("--output", type=str, default="hybrid/output/equator_time_history.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    z = np.load(args.input, allow_pickle=False)
    t = np.asarray(z["t"], dtype=float)
    x = np.asarray(z["x"], dtype=float)  # (n_particle, n_time, 3)

    xx = x[:, :, 0]
    yy = x[:, :, 1]
    zz = x[:, :, 2]
    rr = np.sqrt(xx * xx + yy * yy)

    n_particles, n_time = zz.shape
    eq_mask = np.abs(zz) <= args.z_band

    eq_count = np.sum(eq_mask, axis=0)
    eq_fraction = eq_count / max(n_particles, 1)

    rr_eq = np.where(eq_mask, rr, np.nan)
    mean_r = np.nanmean(rr_eq, axis=0)
    std_r = np.nanstd(rr_eq, axis=0)

    r_edges = np.linspace(0.0, args.r_max, args.bins_r + 1)
    heat = np.zeros((args.bins_r, n_time), dtype=float)
    for j in range(n_time):
        vals = rr[eq_mask[:, j], j]
        heat[:, j], _ = np.histogram(vals, bins=r_edges)

    if len(t) > 1:
        dt = t[1] - t[0]
    else:
        dt = 1.0
    t_edges = np.concatenate([t - 0.5 * dt, [t[-1] + 0.5 * dt]])

    fig = plt.figure(figsize=(11, 10), dpi=170)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.5], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, eq_fraction, color="tab:blue", lw=1.6, label="equatorial fraction")
    ax1.set_ylabel("fraction in |z| band")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.25)
    ax1_t = ax1.twinx()
    ax1_t.plot(t, eq_count, color="tab:orange", lw=1.0, alpha=0.8, label="equatorial count")
    ax1_t.set_ylabel("count")
    ax1.set_title(f"Equatorial time history (|z| <= {args.z_band:g})")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t, mean_r, color="tab:green", lw=1.6, label="mean R in band")
    ax2.fill_between(t, mean_r - std_r, mean_r + std_r, color="tab:green", alpha=0.2, label="±1σ")
    ax2.set_ylabel("R in equatorial band")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    mesh = ax3.pcolormesh(t_edges, r_edges, heat, shading="auto", cmap="magma")
    ax3.set_xlabel("time")
    ax3.set_ylabel("R in equatorial band")
    ax3.set_ylim(0.0, args.r_max)
    ax3.set_title("Equatorial radial density vs time")
    cbar = fig.colorbar(mesh, ax=ax3)
    cbar.set_label("particle count / bin")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    print(f"saved: {out}")
    print(f"n_particles={n_particles} n_time={n_time}")
    print(f"mean equatorial fraction={np.mean(eq_fraction):.3f}")


if __name__ == "__main__":
    main()
