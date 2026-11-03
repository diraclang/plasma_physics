from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dipole_test.simulate import DipoleRunConfig, run_single_particle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dipole test-particle trajectory.")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--q-over-m", type=float, default=1.0)
    p.add_argument("--dipole-scale", type=float, default=1.0, help="Scale factor for dipole moment magnitude")
    p.add_argument("--with-corotation", action="store_true")
    p.add_argument("--omega", type=float, default=0.05, help="Omega_z for corotation")
    p.add_argument("--output", type=str, default="hybrid/output/trajectory.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DipoleRunConfig(
        steps=args.steps,
        dt=args.dt,
        q_over_m=args.q_over_m,
        dipole_moment=(0.0, 0.0, args.dipole_scale),
        with_corotation=args.with_corotation,
        omega_vec=(0.0, 0.0, args.omega),
    )
    result = run_single_particle(cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = result["x"]
    v = result["v"]
    b = result["B"]
    e = result["E"]
    rows = np.column_stack(
        [
            result["t"],
            x[:, 0],
            x[:, 1],
            x[:, 2],
            v[:, 0],
            v[:, 1],
            v[:, 2],
            b[:, 0],
            b[:, 1],
            b[:, 2],
            e[:, 0],
            e[:, 1],
            e[:, 2],
            result["K"],
            result["mu"],
        ]
    )
    header = "t,x,y,z,vx,vy,vz,Bx,By,Bz,Ex,Ey,Ez,K,mu"
    np.savetxt(out_path, rows, delimiter=",", header=header, comments="")

    k0, k1 = result["K"][0], result["K"][-1]
    mu0, mu1 = result["mu"][0], result["mu"][-1]
    print(f"saved: {out_path}")
    print(
        f"steps={args.steps} dt={args.dt} corotation={args.with_corotation} "
        f"dipole_scale={args.dipole_scale}"
    )
    print(f"K drift: {(k1 - k0) / max(abs(k0), 1e-12):+.3e}")
    print(f"mu drift: {(mu1 - mu0) / max(abs(mu0), 1e-12):+.3e}")


if __name__ == "__main__":
    main()
