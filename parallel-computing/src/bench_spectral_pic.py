"""Benchmark the GPU spectral PIC step in the target regime (default 200k particles/species, ~1000 modes).

    PYTHONPATH=. python -m src.bench_spectral_pic
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from . import spectral_kernels as sk
from .spectral_pic_gpu import PICConfig, Species, SpectralPIC


def timed(fn, reps: int = 11) -> float:
    """median seconds"""
    fn()
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def make(n: int, modes, seed: int = 0) -> SpectralPIC:
    """Neutral thermal two-species plasma, uniform B_eq, small transverse seed fields."""
    cfg = PICConfig(L=(256.0, 256.0, 256.0), modes=modes, dt=0.05, c=5.0, eps0=0.04, B_eq=(0.0, 4.0, -4.0))
    rng = np.random.default_rng(seed)
    sps = []
    for q, m, vth in ((+0.4, 4.0, 1.0), (-0.4, 1.0, 2.0)):
        sp = Species(q=q, m=m, weight=1.0)
        pos = rng.uniform(0, 256, (3, n))
        vel = rng.normal(0, vth, (3, n))
        sp.state = tuple(mx.array(c.astype(np.float32)) for c in (*pos, *vel))
        sps.append(sp)
    A, B, C = cfg.shape
    amp = 1e-3 * (rng.normal(size=(3, A, B, C)) + 1j * rng.normal(size=(3, A, B, C)))
    return SpectralPIC(cfg, sps, E_T0=amp, B0=amp)


def stage_times(pic: SpectralPIC) -> dict[str, float]:
    """Time each stage of one step in isolation (each includes its own sync)."""
    sp0 = pic.species[0]
    dt = pic.cfg.dt
    rho = sk.deposit(sp0.state, sp0.rho_params, sp0.plan_rho)
    mx.eval(rho)
    Bnext, Bn = pic._faraday(pic.Bh, pic.ET, dt)
    table, wEL = pic._fields(rho, pic.ET, Bn)
    J = sk.deposit(sp0.state, sp0.J_params, sp0.plan_J)
    mx.eval(Bnext, Bn, table, wEL, J)
    A, B, C = pic.shape
    out = {}
    out["rho deposit (1 species)"] = timed(lambda: mx.eval(sk.deposit(sp0.state, sp0.rho_params, sp0.plan_rho)))
    out["Faraday+Poisson+table"] = timed(lambda: mx.eval(*pic._faraday(pic.Bh, pic.ET, dt), *pic._fields(rho, pic.ET, Bn)))
    out["gather+Boris push (1 species)"] = timed(lambda: mx.eval(*sk.gather_push(sp0.state, table, sp0.gather_params, pic.shape)))
    out["J deposit (1 species)"] = timed(lambda: mx.eval(sk.deposit(sp0.state, sp0.J_params, sp0.plan_J)))
    out["J_T projection + Ampere"] = timed(lambda: mx.eval(pic._ampere(pic.ET, Bnext, J.reshape(3, 2, A, B, C), dt)))
    return out


def step_time(pic: SpectralPIC, nsteps: int = 20, diag_every: int = 0, reps: int = 5) -> float:
    pic.run(3, eval_every=3, diag_every=diag_every)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        pic.run(nsteps, eval_every=nsteps, diag_every=diag_every)
        ts.append((time.perf_counter() - t) / nsteps)
        pic.diag.clear()
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="particles per species")
    ap.add_argument("--modes", type=int, nargs=3, default=[19, 3, 3], help="hx hy hz of the headline run (19,3,3 -> 980 stored modes)")
    ap.add_argument("--cpu-sec-per-step", type=float, default=None, help="legacy CPU seconds/step at the same N, for the speed-up line")
    ap.add_argument("--skip-scaling", action="store_true", help="skip the mode-count sweep")
    args = ap.parse_args()

    modes = tuple(args.modes)
    pic = make(args.n, modes)
    A, B, C = pic.shape
    print(f"N = {args.n} per species (2 species), modes (hx,hy,hz) = {modes}: {A*B*C} stored half-space modes "
          f"({(2*modes[0]+1)*B*C} in the full symmetric set)\n")

    print("== per-stage cost (each stage timed alone with its own sync, after a GPU warm-up) ==")
    pic.run(30, eval_every=10)  # let the GPU clock up; otherwise the first stage timed is pessimistic
    st = stage_times(pic)
    for k, v in st.items():
        print(f"   {k:<32} {v * 1e3:8.3f} ms")
    est = (2 * st["rho deposit (1 species)"] + st["Faraday+Poisson+table"] + 2 * st["gather+Boris push (1 species)"]
           + 2 * st["J deposit (1 species)"] + st["J_T projection + Ampere"])
    print(f"   sum for a full step (2 species)  {est * 1e3:8.3f} ms   (upper bound: every stage here pays its own sync)")

    print("\n== whole step, kernels queued, one sync per 20 steps ==")
    t0 = step_time(pic)
    t1 = step_time(pic, diag_every=1)
    t10 = step_time(pic, diag_every=10)
    print(f"   no diagnostics          : {t0 * 1e3:8.3f} ms/step")
    print(f"   energies every 10 steps : {t10 * 1e3:8.3f} ms/step")
    print(f"   energies every step     : {t1 * 1e3:8.3f} ms/step")
    print(f"   -> {2 * args.n / t0 / 1e6:.1f} M particle-steps/s")

    if not args.skip_scaling:
        print("\n== scaling with mode count (N fixed, whole step, no diagnostics) ==")
        print(f"{'modes (hx,hy,hz)':>18} {'stored K':>9} {'ms/step':>9} {'us per K':>9}")
        for mm in ((3, 1, 1), (8, 2, 2), (12, 3, 3), (19, 3, 3), (16, 5, 5)):
            p = make(args.n, mm)
            a, b, c = p.shape
            t = step_time(p, reps=3)
            print(f"{str(mm):>18} {a*b*c:>9} {t * 1e3:>9.3f} {t * 1e6 / (a*b*c):>9.2f}")

    if args.cpu_sec_per_step:
        print(f"\n== vs legacy CPU: {args.cpu_sec_per_step:.3f} s/step -> speed-up {args.cpu_sec_per_step / t0:.0f}x "
              f"(headline GPU config above, no diagnostics) ==")


if __name__ == "__main__":
    main()
