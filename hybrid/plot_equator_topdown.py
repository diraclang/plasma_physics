from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-down equatorial projection diagnostics.")
    p.add_argument("--input", type=str, default="hybrid/output/ensemble.npz")
    p.add_argument("--z-band", type=float, default=0.3, help="Use points with |z| <= z-band")
    p.add_argument("--r-max", type=float, default=8.0)
    p.add_argument("--out-png", type=str, default="hybrid/output/equator_topdown.png")
    p.add_argument("--out-gif", type=str, default="hybrid/output/equator_topdown.gif")
    p.add_argument("--gif-frames", type=int, default=60)
    p.add_argument("--gif-fps", type=int, default=8)
    return p.parse_args()


def circular_mean(phi: np.ndarray, axis: int) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(phi), axis=axis), np.mean(np.cos(phi), axis=axis))


def main() -> None:
    args = parse_args()
    z = np.load(args.input, allow_pickle=False)
    t = np.asarray(z["t"], dtype=float)
    x = np.asarray(z["x"], dtype=float)  # (n_particle, n_time, 3)

    xx = x[:, :, 0]
    yy = x[:, :, 1]
    zz = x[:, :, 2]
    rr = np.sqrt(xx * xx + yy * yy)
    phi = np.arctan2(yy, xx)

    eq = np.abs(zz) <= args.z_band

    # Prepare pooled equatorial points for the top-down density panel.
    xp = xx[eq]
    yp = yy[eq]

    # Build an unwrapped mean azimuth history to reveal net sweep direction/rate.
    phi_mean = np.full(phi.shape[1], np.nan, dtype=float)
    for j in range(phi.shape[1]):
        mask = eq[:, j]
        if np.any(mask):
            phi_mean[j] = circular_mean(phi[mask, j], axis=0)
    valid = np.isfinite(phi_mean)
    phi_unwrapped = np.full_like(phi_mean, np.nan)
    if np.any(valid):
        phi_unwrapped[valid] = np.unwrap(phi_mean[valid])

    # Static summary figure.
    fig = plt.figure(figsize=(12, 10), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.2)

    ax0 = fig.add_subplot(gs[0, 0])
    h = ax0.hist2d(xp, yp, bins=140, range=[[-args.r_max, args.r_max], [-args.r_max, args.r_max]], cmap="viridis")
    fig.colorbar(h[3], ax=ax0, label="counts")
    ax0.set_title(f"Top-down equatorial density (|z| <= {args.z_band:g})")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_aspect("equal")
    ax0.set_xlim(-args.r_max, args.r_max)
    ax0.set_ylim(-args.r_max, args.r_max)

    ax1 = fig.add_subplot(gs[0, 1])
    idxs = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    labels = [f"t={t[i]:.2f}" for i in idxs]
    for c, i, lbl in zip(colors, idxs, labels):
        mask = eq[:, i]
        ax1.scatter(xx[mask, i], yy[mask, i], s=14, alpha=0.65, color=c, label=lbl)
    ax1.set_title("Equatorial points at selected times")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim(-args.r_max, args.r_max)
    ax1.set_ylim(-args.r_max, args.r_max)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t[valid], np.degrees(phi_unwrapped[valid]), lw=1.6, color="tab:purple")
    ax2.set_title("Mean equatorial azimuth angle vs time (unwrapped)")
    ax2.set_xlabel("time")
    ax2.set_ylabel("mean azimuth (deg)")
    ax2.grid(alpha=0.25)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # Optional GIF: top-down equatorial scatter vs time.
    out_gif = Path(args.out_gif)
    fig2, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    writer = PillowWriter(fps=args.gif_fps)
    writer.setup(fig2, str(out_gif), dpi=160)

    frame_ids = np.linspace(0, len(t) - 1, args.gif_frames, dtype=int)
    for i in frame_ids:
        ax.clear()
        mask = eq[:, i]
        ax.scatter(xx[mask, i], yy[mask, i], s=16, alpha=0.75, color="tab:blue")
        circle = plt.Circle((0.0, 0.0), 1.0, fill=False, lw=1.2, color="black", alpha=0.7)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-args.r_max, args.r_max)
        ax.set_ylim(-args.r_max, args.r_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Pole-down equatorial projection, t={t[i]:.3f}")
        ax.grid(alpha=0.2)
        writer.grab_frame()
    writer.finish()
    plt.close(fig2)

    print(f"saved: {out_png}")
    print(f"saved: {out_gif}")
    print(f"equatorial sample fraction: {np.mean(eq):.3f}")


if __name__ == "__main__":
    main()
