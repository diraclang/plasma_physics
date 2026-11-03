from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot self-consistent meridian evolution (density + field lines).")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out-prefix", type=str, default="hybrid/output/sc_meridian")
    p.add_argument("--max-density", type=float, default=None, help="Colorbar max for density; auto if omitted")
    p.add_argument("--gif-fps", type=int, default=6)
    p.add_argument("--plot-every", type=int, default=1, help="Use every Nth stored snapshot")
    p.add_argument("--r-max", type=float, default=None, help="Optional x-limit for meridian panel")
    p.add_argument("--z-max", type=float, default=None, help="Optional z-limit for meridian panel")
    return p.parse_args()


def _build_meridian_field(
    n_rz: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.concatenate([-r_grid[:0:-1], r_grid])
    nx = x_grid.size
    nz = z_grid.size

    bx = np.zeros((nz, nx), dtype=float)
    bz_plane = np.zeros((nz, nx), dtype=float)
    n_plane = np.zeros((nz, nx), dtype=float)

    for ix, x in enumerate(x_grid):
        sign = -1.0 if x < 0.0 else 1.0
        ridx = int(np.argmin(np.abs(r_grid - abs(x))))
        bx[:, ix] = sign * br[ridx, :]
        bz_plane[:, ix] = -bz[ridx, :]
        n_plane[:, ix] = n_rz[ridx, :]

    xx, zz = np.meshgrid(x_grid, z_grid)
    return x_grid, z_grid, xx, zz, n_plane, bx, bz_plane


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=False)

    required = ["snap_t_idx", "snap_n", "snap_br", "snap_bz", "r_grid", "z_grid", "t"]
    for key in required:
        if key not in data.files or data[key].size == 0:
            raise ValueError(f"Missing required self-consistent snapshot field: {key}")

    snap_t_idx = np.asarray(data["snap_t_idx"], dtype=int)
    snap_n = np.asarray(data["snap_n"], dtype=float)
    snap_br = np.asarray(data["snap_br"], dtype=float)
    snap_bz = np.asarray(data["snap_bz"], dtype=float)
    r_grid = np.asarray(data["r_grid"], dtype=float)
    z_grid = np.asarray(data["z_grid"], dtype=float)
    t = np.asarray(data["t"], dtype=float)

    stride = max(args.plot_every, 1)
    idx = np.arange(0, snap_t_idx.size, stride)
    if idx[-1] != snap_t_idx.size - 1:
        idx = np.append(idx, snap_t_idx.size - 1)

    snap_t_idx = snap_t_idx[idx]
    snap_n = snap_n[idx]
    snap_br = snap_br[idx]
    snap_bz = snap_bz[idx]

    positive = snap_n[snap_n > 0.0]
    if positive.size == 0:
        raise ValueError("No positive density values found in the snapshot data.")

    dens_min = float(np.min(positive))
    dens_max = args.max_density
    if dens_max is None:
        dens_max = float(np.percentile(positive, 99.9))
        if not np.isfinite(dens_max) or dens_max <= dens_min:
            dens_max = float(np.max(positive))
    dens_max = max(float(dens_max), dens_min * 10.0)
    norm = matplotlib.colors.LogNorm(vmin=max(dens_min, 1e-8), vmax=dens_max)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_name(out_prefix.name + "_panels.png")
    gif_path = out_prefix.with_name(out_prefix.name + "_evolution.gif")

    # Static panel of selected times.
    panel_count = min(6, snap_t_idx.size)
    panel_ids = np.linspace(0, snap_t_idx.size - 1, panel_count).astype(int)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs = axs.ravel()

    for ax, sid in zip(axs, panel_ids, strict=False):
        n2d = snap_n[sid]
        br2d = snap_br[sid]
        bz2d = snap_bz[sid]
        x_grid, zvec, xx, zz, n_plane, bx, bz_plane = _build_meridian_field(n2d, br2d, bz2d, r_grid, z_grid)

        masked = np.ma.masked_where(n_plane <= 0.0, n_plane)
        pcm = ax.pcolormesh(xx, zz, masked, shading="auto", cmap="magma", norm=norm)
        ax.streamplot(x_grid, zvec, bx, bz_plane, color="w", density=1.2, linewidth=0.7, arrowsize=0.8)
        ts = t[snap_t_idx[sid]]
        ax.set_title(f"t={ts:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        if args.r_max is not None:
            ax.set_xlim(-args.r_max, args.r_max)
        if args.z_max is not None:
            ax.set_ylim(-args.z_max, args.z_max)

    for ax in axs[len(panel_ids) :]:
        ax.axis("off")

    cbar = fig.colorbar(pcm, ax=axs.tolist(), shrink=0.9)
    cbar.set_label("Ion density (axisymmetric, meridian)")
    fig.suptitle("Self-consistent settling: density + total B field lines")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    # GIF evolution.
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)

    def draw_frame(sid: int) -> None:
        ax2.clear()
        n2d = snap_n[sid]
        br2d = snap_br[sid]
        bz2d = snap_bz[sid]
        x_grid, zvec, xx, zz, n_plane, bx, bz_plane = _build_meridian_field(n2d, br2d, bz2d, r_grid, z_grid)
        masked = np.ma.masked_where(n_plane <= 0.0, n_plane)
        ax2.pcolormesh(xx, zz, masked, shading="auto", cmap="magma", norm=norm)
        ax2.streamplot(x_grid, zvec, bx, bz_plane, color="w", density=1.2, linewidth=0.7, arrowsize=0.8)
        ts = t[snap_t_idx[sid]]
        ax2.set_title(f"t={ts:.3f}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        if args.r_max is not None:
            ax2.set_xlim(-args.r_max, args.r_max)
        if args.z_max is not None:
            ax2.set_ylim(-args.z_max, args.z_max)

    writer = PillowWriter(fps=max(args.gif_fps, 1))
    with writer.saving(fig2, str(gif_path), dpi=130):
        for sid in range(snap_t_idx.size):
            draw_frame(sid)
            writer.grab_frame()

    plt.close(fig2)
    print(f"saved: {png_path}")
    print(f"saved: {gif_path}")


if __name__ == "__main__":
    main()
