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
    p.add_argument("--density-b", action="store_true", help="Also write a density + |B| figure (field lines colored by log10|B|)")
    p.add_argument("--toroidal-b", type=float, default=0.0, help="Background toroidal B0 used in the run (for |B| in --density-b)")
    p.add_argument("--toroidal-r0", type=float, default=1.0, help="Toroidal core radius used in the run (for |B| in --density-b)")
    return p.parse_args()


def _meridian_x(r_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evenly spaced mirrored x axis, nearest r index per x, and mask of the empty inner gap.

    For r_min = 0 this is the plain [-r_grid[:0:-1], r_grid] axis.  For an annulus
    (r_min > 0) the axis is extended inward to x = 0 so it stays evenly spaced (needed by
    streamplot); points with |x| < r_min are flagged in the returned gap mask.
    """
    dr = float(r_grid[1] - r_grid[0])
    k = int(round(float(r_grid[0]) / dr)) if r_grid[0] > 0.0 else 0
    r_ext = np.clip(r_grid[0] + dr * np.arange(-k, r_grid.size), 0.0, None)
    x_grid = np.concatenate([-r_ext[:0:-1], r_ext])
    ridx = np.abs(r_grid[None, :] - np.abs(x_grid)[:, None]).argmin(axis=1)
    gap = np.abs(x_grid) < float(r_grid[0]) - 0.5 * dr
    return x_grid, ridx, gap


def _build_meridian_field(
    n_rz: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid, ridx, gap = _meridian_x(r_grid)
    sign = np.where(x_grid < 0.0, -1.0, 1.0)

    bx = sign[None, :] * br[ridx, :].T
    # In the meridian x-z plane, Bz keeps the same sign as cylindrical bz.
    bz_plane = bz[ridx, :].T.copy()
    n_plane = n_rz[ridx, :].T.copy()
    if gap.any():
        n_plane[:, gap] = 0.0
        mask = np.broadcast_to(gap[None, :], bx.shape)
        bx = np.ma.array(bx, mask=mask)
        bz_plane = np.ma.array(bz_plane, mask=mask)

    xx, zz = np.meshgrid(x_grid, z_grid)
    return x_grid, z_grid, xx, zz, n_plane, bx, bz_plane


def _write_density_b_figure(
    png_path: Path,
    ids: np.ndarray,
    snap_t_idx: np.ndarray,
    snap_n: np.ndarray,
    snap_br: np.ndarray,
    snap_bz: np.ndarray,
    snap_dbphi: np.ndarray | None,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    t_snap: np.ndarray,
    norm: matplotlib.colors.Normalize,
    args: argparse.Namespace,
) -> None:
    r0 = max(float(args.toroidal_r0), 1e-6)
    rr = r_grid[:, None]
    bphi0 = np.where(rr < r0, args.toroidal_b * rr / r0, args.toroidal_b * r0 / np.maximum(rr, 1e-9))
    bphi0 = bphi0 * np.ones((1, z_grid.size))

    x_grid, ridx_x, gap_x = _meridian_x(r_grid)
    planes = []
    for sid in ids:
        dbphi = snap_dbphi[sid] if snap_dbphi is not None else 0.0
        bmag_rz = np.sqrt(snap_br[sid] ** 2 + snap_bz[sid] ** 2 + (bphi0 + dbphi) ** 2)
        bmag_plane = bmag_rz[ridx_x, :].T
        planes.append(np.log10(np.maximum(bmag_plane, 1e-12)))

    xlim = args.r_max if args.r_max is not None else float(r_grid[-1])
    zlim = args.z_max if args.z_max is not None else float(z_grid[-1])
    xx, zz = np.meshgrid(x_grid, z_grid)
    shown = (np.abs(xx) <= xlim) & (np.abs(zz) <= zlim) & ~gap_x[None, :]
    allvals = np.concatenate([p[shown] for p in planes])
    bnorm = matplotlib.colors.Normalize(vmin=float(np.percentile(allvals, 2)), vmax=float(np.percentile(allvals, 98)))

    fig, axs = plt.subplots(1, len(ids), figsize=(6.4 * len(ids), 6.2), constrained_layout=True, squeeze=False)
    pcm = None
    strm = None
    for ax, sid, logb in zip(axs[0], ids, planes, strict=False):
        _x, zvec, xx, zz, n_plane, bx, bz_plane = _build_meridian_field(snap_n[sid], snap_br[sid], snap_bz[sid], r_grid, z_grid)
        masked = np.ma.masked_where(n_plane <= 0.0, n_plane)
        pcm = ax.pcolormesh(xx, zz, masked, shading="auto", cmap="magma", norm=norm)
        strm = ax.streamplot(x_grid, zvec, bx, bz_plane, color=logb, cmap="cool", norm=bnorm, density=1.4, linewidth=0.9, arrowsize=0.8)
        ax.set_title(f"t={t_snap[sid]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-zlim, zlim)
        ax.set_aspect("equal")
    fig.colorbar(pcm, ax=axs.ravel().tolist(), shrink=0.8, location="right", label="Ion density")
    fig.colorbar(strm.lines, ax=axs.ravel().tolist(), shrink=0.8, location="right", label="log10 |B| (field lines, includes B_phi)")
    fig.suptitle("Meridian plane: ion density (color map) + magnetic field lines colored by |B|")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


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
    if "hist_step" in data.files and data["hist_step"].size == t.size:
        # Particle histories were thinned: time is linear in the step number.
        t_snap = np.interp(snap_t_idx, np.asarray(data["hist_step"], dtype=float), t)
    else:
        t_snap = t[snap_t_idx]

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
        ts = t_snap[sid]
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
        ts = t_snap[sid]
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

    if args.density_b:
        dbphi_all = None
        if "snap_dbphi" in data.files and data["snap_dbphi"].size > 0:
            dbphi_all = np.asarray(data["snap_dbphi"], dtype=float)[idx]
        db_ids = np.unique(np.linspace(0, snap_t_idx.size - 1, 3).astype(int))
        db_path = out_prefix.with_name(out_prefix.name + "_density_B.png")
        _write_density_b_figure(
            db_path, db_ids, snap_t_idx, snap_n, snap_br, snap_bz, dbphi_all, r_grid, z_grid, t_snap, norm, args
        )
        print(f"saved: {db_path}")


if __name__ == "__main__":
    main()
