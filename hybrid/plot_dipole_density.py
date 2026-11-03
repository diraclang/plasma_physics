from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot density maps from dipole ensemble output.")
    p.add_argument("--input", type=str, default="hybrid/output/ensemble.npz")
    p.add_argument("--phi-center", type=float, default=0.0, help="Center angle in degrees")
    p.add_argument("--phi-width", type=float, default=10.0, help="Slice width in degrees")
    p.add_argument("--phi-min", type=float, default=0.0, help="Collapsed range min (deg)")
    p.add_argument("--phi-max", type=float, default=30.0, help="Collapsed range max (deg)")
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--z-max", type=float, default=6.0)
    p.add_argument("--bins-r", type=int, default=160)
    p.add_argument("--bins-z", type=int, default=180)
    p.add_argument("--output-prefix", type=str, default="hybrid/output/density")
    return p.parse_args()


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 360.0)


def angle_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(wrap_deg(a) - wrap_deg(b))
    return np.minimum(d, 360.0 - d)


def in_range_wrapped(phi: np.ndarray, amin: float, amax: float) -> np.ndarray:
    phi = wrap_deg(phi)
    amin = float(wrap_deg(np.array([amin]))[0])
    amax = float(wrap_deg(np.array([amax]))[0])
    if amin <= amax:
        return (phi >= amin) & (phi <= amax)
    return (phi >= amin) | (phi <= amax)


def dipole_field_lines(ax: plt.Axes, l_values: list[float]) -> None:
    theta = np.linspace(1e-3, np.pi - 1e-3, 500)
    for l in l_values:
        r = l * np.sin(theta) ** 2
        rr = r * np.sin(theta)
        zz = r * np.cos(theta)
        ax.plot(rr, zz, color="white", lw=0.8, alpha=0.7)


def density_panel(
    rvals: np.ndarray,
    zvals: np.ndarray,
    args: argparse.Namespace,
    title: str,
    output_path: Path,
) -> None:
    r_edges = np.linspace(0.0, args.r_max, args.bins_r + 1)
    z_edges = np.linspace(-args.z_max, args.z_max, args.bins_z + 1)
    h, _, _ = np.histogram2d(rvals, zvals, bins=[r_edges, z_edges])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    im = ax.pcolormesh(r_edges, z_edges, h.T, shading="auto", cmap="viridis")
    dipole_field_lines(ax, [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ax.set_xlabel("R = sqrt(x^2 + y^2)")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_xlim(0.0, args.r_max)
    ax.set_ylim(-args.z_max, args.z_max)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("counts per bin")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    z = np.load(args.input, allow_pickle=False)
    x = np.asarray(z["x"])  # (n_particle, n_time, 3)

    xyz = x.reshape(-1, 3)
    xx = xyz[:, 0]
    yy = xyz[:, 1]
    zz = xyz[:, 2]
    rr = np.sqrt(xx * xx + yy * yy)
    phi = wrap_deg(np.degrees(np.arctan2(yy, xx)))

    dphi = angle_distance_deg(phi, args.phi_center)
    m_slice = dphi <= 0.5 * args.phi_width
    m_range = in_range_wrapped(phi, args.phi_min, args.phi_max)

    prefix = Path(args.output_prefix)
    f1 = prefix.with_name(prefix.name + "_slice.png")
    f2 = prefix.with_name(prefix.name + "_collapsed.png")

    density_panel(
        rr[m_slice],
        zz[m_slice],
        args,
        title=f"Density in azimuth slice: phi={args.phi_center:.1f}±{0.5*args.phi_width:.1f} deg",
        output_path=f1,
    )
    density_panel(
        rr[m_range],
        zz[m_range],
        args,
        title=f"Density collapsed over phi in [{args.phi_min:.1f}, {args.phi_max:.1f}] deg",
        output_path=f2,
    )

    print(f"saved: {f1}")
    print(f"saved: {f2}")
    print(f"slice samples: {int(np.sum(m_slice))}")
    print(f"collapsed samples: {int(np.sum(m_range))}")


if __name__ == "__main__":
    main()
