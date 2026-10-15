"""Boris push in a uniform B = (0, 0, B0) on the Apple GPU via MLX.

Design (each point was measured):
  * struct-of-arrays state (six flat float32 arrays) instead of an (N, 3) array,
    so every kernel reads/writes contiguous memory;
  * the step is wrapped in mx.compile so all element-wise ops fuse into one kernel;
  * no mx.eval / host copy inside the time loop (MLX is lazy; only evaluate at
    the end or at diagnostic points);
  * pure magnetic field: no E half-kick, so kinetic energy is conserved.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

import mlx.core as mx


@dataclass
class Config:
    B0: float = 1.0
    dt: float = 0.01
    box_size: float = 1.0
    q_over_m: float = 1.0
    thermal_speed: float = 0.2
    seed: int = 7


def initial_state(n: int, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    pos = rng.uniform(0.0, cfg.box_size, size=(n, 3))
    vel = rng.normal(0.0, cfg.thermal_speed, size=(n, 3))
    return pos, vel


# ----------------------------------------------------------------------------
# CPU references (float64)
# ----------------------------------------------------------------------------
def numpy_reference(pos: np.ndarray, vel: np.ndarray, cfg: Config, steps: int):
    """Generic Boris with np.cross on (N, 3) float64 arrays (independent of the GPU code)."""
    p = pos.astype(np.float64).copy()
    v = vel.astype(np.float64).copy()
    t = 0.5 * cfg.q_over_m * cfg.dt * np.array([0.0, 0.0, cfg.B0])
    s = 2.0 * t / (1.0 + float(t @ t))
    for _ in range(steps):
        v_prime = v + np.cross(v, t)
        v = v + np.cross(v_prime, s)
        p = np.mod(p + cfg.dt * v, cfg.box_size)
    return p, v


# ----------------------------------------------------------------------------
# GPU solver
# ----------------------------------------------------------------------------
class GPUMagneticSolver:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, cfg: Config) -> None:
        mx.set_default_device(mx.gpu)
        self.cfg = cfg
        self.device = str(mx.default_device())
        t = 0.5 * cfg.q_over_m * cfg.dt * cfg.B0
        s = 2.0 * t / (1.0 + t * t)
        dt, box = cfg.dt, cfg.box_size

        def step(x, y, z, vx, vy, vz):
            vpx = vx + vy * t
            vpy = vy - vx * t
            nvx = vx + vpy * s
            nvy = vy - vpx * s
            return (
                (x + dt * nvx) % box,
                (y + dt * nvy) % box,
                (z + dt * vz) % box,
                nvx,
                nvy,
                vz,
            )

        self._step = mx.compile(step)
        cols = [pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2]]
        self.state = tuple(mx.array(np.ascontiguousarray(c), dtype=mx.float32) for c in cols)
        mx.eval(*self.state)

    def advance(self, steps: int) -> None:
        """Queue `steps` updates and evaluate once at the end."""
        state = self.state
        for _ in range(steps):
            state = self._step(*state)
        mx.eval(*state)
        self.state = state

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        arrs = [np.asarray(a, dtype=np.float64) for a in self.state]
        return np.stack(arrs[:3], axis=1), np.stack(arrs[3:], axis=1)

    def kinetic_energy(self) -> float:
        vx, vy, vz = self.state[3:]
        return float(0.5 * mx.sum(vx * vx + vy * vy + vz * vz))


# ----------------------------------------------------------------------------
# Validation + benchmark
# ----------------------------------------------------------------------------
def periodic_dist(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, box - d)


def validate(n: int, steps: int, cfg: Config) -> dict:
    pos, vel = initial_state(n, cfg)
    ref_p, ref_v = numpy_reference(pos, vel, cfg, steps)
    solver = GPUMagneticSolver(pos, vel, cfg)
    ke0 = float(0.5 * np.sum(vel * vel))
    solver.advance(steps)
    p, v = solver.to_numpy()
    ke = solver.kinetic_energy()
    return {
        "device": solver.device,
        "max_pos_err": float(np.max(periodic_dist(p, ref_p, cfg.box_size))),
        "max_vel_err": float(np.max(np.abs(v - ref_v))),
        "ke_rel_drift_gpu": (ke - ke0) / ke0,
        "ke_rel_drift_ref": (float(0.5 * np.sum(ref_v * ref_v)) - ke0) / ke0,
    }


def time_gpu(n: int, steps: int, cfg: Config, reps: int = 5) -> float:
    pos, vel = initial_state(n, cfg)
    solver = GPUMagneticSolver(pos, vel, cfg)
    solver.advance(1)  # warm-up / compile
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        solver.advance(steps)
        best = min(best, time.perf_counter() - t0)
    return best / steps


def time_numpy(n: int, steps: int, cfg: Config) -> float:
    pos, vel = initial_state(n, cfg)
    t0 = time.perf_counter()
    numpy_reference(pos, vel, cfg, steps)
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser(description="MLX GPU Boris push, uniform B")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--B0", type=float, default=1.0)
    ap.add_argument("--validate-particles", type=int, default=100_000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 4_000_000])
    args = ap.parse_args()
    cfg = Config(B0=args.B0, dt=args.dt)

    v = validate(args.validate_particles, args.steps, cfg)
    print("validation vs float64 NumPy reference:", v)

    print(f"\n{'N':>10} {'numpy f64 ms/step':>18} {'gpu ms/step':>12} {'speedup':>8}")
    for n in args.sizes:
        cpu = time_numpy(n, min(args.steps, 10), cfg)
        gpu = time_gpu(n, args.steps, cfg)
        print(f"{n:>10} {1e3 * cpu:>18.3f} {1e3 * gpu:>12.3f} {cpu / gpu:>7.1f}x")


if __name__ == "__main__":
    main()
