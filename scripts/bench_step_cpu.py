"""Time the legacy CPU step (coupled_loop.run_coupled_loop) at a given particle count and mode set.

    PYTHONPATH=src python scripts/bench_step_cpu.py --n 200000 --mmax 3 --nmax 1

Per-step time = (T(k2 steps) - T(k1 steps)) / (k2 - k1), which removes particle loading / setup.
Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from config import SimulationConfig, mode_range
from coupled_loop import run_coupled_loop
from particles import _mean_density


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="macro-particles per species")
    ap.add_argument("--mmax", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=1)
    ap.add_argument("--k1", type=int, default=2)
    ap.add_argument("--k2", type=int, default=6)
    args = ap.parse_args()

    cfg = SimulationConfig(Lx=256.0, Ly=256.0, Lz=256.0, lam=20.0, B0=4.0, Bg=4.0, nx=256, dt=0.05,
                           modes=mode_range(range(-args.mmax, args.mmax + 1), range(-args.nmax, args.nmax + 1),
                                            seed_amplitude=1.0e-3))
    cfg.particles_per_cell = max(1, round(args.n / (cfg.nx * _mean_density(cfg))))

    times = {}
    n_part = None
    for k in (args.k1, args.k2):
        t0 = time.perf_counter()
        h = run_coupled_loop(cfg, n_steps=k, paired_load=True)
        times[k] = time.perf_counter() - t0
        n_part = h["species"][0].n
    per_step = (times[args.k2] - times[args.k1]) / (args.k2 - args.k1)
    n_modes = len(h["modes"])
    print(json.dumps({"n_per_species": int(n_part), "n_modes_halfspace_incl_mean": n_modes, "nx": cfg.nx,
                      "sec_per_step": per_step, "dt": float(h["dt"]), "steps_timed": args.k2 - args.k1}))


if __name__ == "__main__":
    main()
