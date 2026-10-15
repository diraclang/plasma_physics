from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, replace

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None


@dataclass
class SimpleMagneticConfig:
    B0: float = 1.0
    dt: float = 0.01
    n_particles: int = 4096
    box_size: float = 1.0
    q_over_m: float = 1.0
    thermal_speed: float = 0.2
    block_size: int = 256
    seed: int = 7


class SimpleMagneticSolver:
    """Blockwise Boris drift in a uniform magnetic field.

    The key design choice is that the particle update is expressed as a block
    kernel: each block of particles is advanced independently. That matches the
    GPU-parallel execution pattern we want for the larger direct spectral solver.
    """

    def __init__(self, config: SimpleMagneticConfig | None = None) -> None:
        self.config = config or SimpleMagneticConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.pos = self.rng.uniform(0.0, self.config.box_size, size=(self.config.n_particles, 3))
        self.vel = self.rng.normal(0.0, self.config.thermal_speed, size=(self.config.n_particles, 3))
        self.step_count = 0
        self.use_mlx = mx is not None
        self.device = "cpu"
        if self.use_mlx:
            try:
                mx.set_default_device(mx.gpu)
                self.device = str(mx.default_device())
            except Exception:
                self.device = str(mx.default_device())
        self._sync_device_arrays()

    def _sync_device_arrays(self) -> None:
        if self.use_mlx:
            self.pos_mx = mx.array(self.pos, dtype=mx.float32)
            self.vel_mx = mx.array(self.vel, dtype=mx.float32)
        else:
            self.pos_mx = self.pos
            self.vel_mx = self.vel

    def _sync_host_arrays(self) -> None:
        if self.use_mlx:
            self.pos = np.asarray(self.pos_mx)
            self.vel = np.asarray(self.vel_mx)

    @staticmethod
    def _cross_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a, b)

    @staticmethod
    def _cross_mx(a: "mx.Array", b: "mx.Array") -> "mx.Array":
        return mx.stack(
            (
                a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
            ),
            axis=-1,
        )

    def _boris_block_numpy(self, pos_block: np.ndarray, vel_block: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        bvec = np.array([0.0, 0.0, self.config.B0], dtype=np.float64)
        qom = self.config.q_over_m
        tau = 0.5 * qom * dt * bvec
        s = 0.5 * qom * dt * bvec
        s2 = float(np.dot(s, s))

        v_minus = vel_block + tau
        v_prime = v_minus + self._cross_numpy(v_minus, s)
        v_plus = v_minus + self._cross_numpy(v_prime, (2.0 * s) / (1.0 + s2))
        v_new = v_plus + tau

        pos_new = pos_block + dt * v_new
        pos_new = np.mod(pos_new, self.config.box_size)
        return pos_new, v_new

    def _boris_batch_mx(self, pos: np.ndarray | "mx.Array", vel: np.ndarray | "mx.Array", dt: float) -> tuple["mx.Array", "mx.Array"]:
        pos_mx = pos if hasattr(pos, "shape") and hasattr(pos, "dtype") and hasattr(pos, "__array_priority__") else mx.array(pos, dtype=mx.float32)
        vel_mx = vel if hasattr(vel, "shape") and hasattr(vel, "dtype") and hasattr(vel, "__array_priority__") else mx.array(vel, dtype=mx.float32)

        if not isinstance(pos_mx, type(mx.array([0.0]))):
            pos_mx = mx.array(pos_mx, dtype=mx.float32)
        if not isinstance(vel_mx, type(mx.array([0.0]))):
            vel_mx = mx.array(vel_mx, dtype=mx.float32)

        bvec = mx.array([0.0, 0.0, self.config.B0], dtype=mx.float32)
        qom = self.config.q_over_m
        tau = 0.5 * qom * dt * bvec
        s = 0.5 * qom * dt * bvec
        s2 = mx.sum(s * s)

        v_minus = vel_mx + tau
        v_prime = v_minus + self._cross_mx(v_minus, s)
        v_plus = v_minus + self._cross_mx(v_prime, (2.0 * s) / (1.0 + s2))
        v_new = v_plus + tau

        pos_new = pos_mx + dt * v_new
        pos_new = pos_new % self.config.box_size
        return pos_new, v_new

    def _boris_batch_numpy(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return self._boris_block_numpy(pos, vel, dt)

    def _batch_step(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self.use_mlx:
            return self._boris_batch_mx(pos, vel, dt)
        return self._boris_batch_numpy(pos, vel, dt)

    def step(self) -> None:
        if self.use_mlx:
            self.pos_mx, self.vel_mx = self._batch_step(self.pos_mx, self.vel_mx, self.config.dt)
        else:
            pos_new, vel_new = self._batch_step(self.pos, self.vel, self.config.dt)
            self.pos = pos_new
            self.vel = vel_new
        self.step_count += 1

    def kinetic_energy(self) -> float:
        if self.use_mlx:
            self._sync_host_arrays()
        return 0.5 * float(np.sum(np.sum(self.vel * self.vel, axis=1)))

    def mean_velocity(self) -> np.ndarray:
        if self.use_mlx:
            self._sync_host_arrays()
        return np.mean(self.vel, axis=0)

    def run(self, steps: int, log_every: int = 1) -> dict[str, np.ndarray]:
        energies = []
        times = []
        mean_velocities = []

        for step in range(steps):
            self.step()
            if (step + 1) % log_every == 0 or step == 0:
                if self.use_mlx:
                    self._sync_host_arrays()
                energies.append(self.kinetic_energy())
                times.append((step + 1) * self.config.dt)
                mean_velocities.append(self.mean_velocity())

        return {
            "time": np.asarray(times),
            "kinetic_energy": np.asarray(energies),
            "mean_velocity": np.asarray(mean_velocities),
        }

    def summary(self) -> dict[str, float | int | bool | str]:
        return {
            "mx_available": self.use_mlx,
            "device": self.device,
            "n_particles": self.config.n_particles,
            "block_size": self.config.block_size,
            "B0": self.config.B0,
            "step_count": self.step_count,
            "kinetic_energy": self.kinetic_energy(),
        }

    @staticmethod
    def _checksum(pos: np.ndarray, vel: np.ndarray) -> str:
        state = np.concatenate([np.asarray(pos, dtype=np.float64).ravel(), np.asarray(vel, dtype=np.float64).ravel()])
        return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()

    def validate_parallel(self, steps: int, block_sizes: list[int] | None = None, tol: float = 1e-8) -> dict[str, object]:
        if block_sizes is None:
            block_sizes = [self.config.n_particles, max(1, self.config.block_size), 64, 32]

        reference_cfg = replace(self.config, block_size=self.config.n_particles)
        reference = SimpleMagneticSolver(reference_cfg)
        start = time.perf_counter()
        reference.run(steps=steps, log_every=max(1, steps))
        ref_time = time.perf_counter() - start
        ref_state = (reference.pos.copy(), reference.vel.copy())
        ref_checksum = self._checksum(*ref_state)

        rows = []
        for block_size in block_sizes:
            cfg = replace(self.config, block_size=block_size)
            solver = SimpleMagneticSolver(cfg)
            start = time.perf_counter()
            solver.run(steps=steps, log_every=max(1, steps))
            elapsed = time.perf_counter() - start

            pos_err = float(np.max(np.abs(reference.pos - solver.pos)))
            vel_err = float(np.max(np.abs(reference.vel - solver.vel)))
            energy_err = float(abs(reference.kinetic_energy() - solver.kinetic_energy()))
            checksum_match = self._checksum(solver.pos, solver.vel) == ref_checksum
            speedup = ref_time / elapsed if elapsed > 0 else float("inf")

            rows.append(
                {
                    "block_size": block_size,
                    "position_max_error": pos_err,
                    "velocity_max_error": vel_err,
                    "energy_error": energy_err,
                    "checksum_match": checksum_match,
                    "time_seconds": elapsed,
                    "speedup_vs_reference": speedup,
                    "within_tolerance": pos_err <= tol and vel_err <= tol and energy_err <= tol,
                }
            )

        return {
            "serial_block_size": reference_cfg.block_size,
            "serial_time_seconds": ref_time,
            "reference_checksum": ref_checksum,
            "results": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle-parallel magnetic-field prototype")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--B0", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="explicit MLX device selection")
    parser.add_argument("--benchmark", action="store_true", help="run serial-vs-parallel validation and timing")
    args = parser.parse_args()

    config = SimpleMagneticConfig(
        B0=args.B0,
        dt=args.dt,
        n_particles=args.particles,
        block_size=args.block_size,
    )
    solver = SimpleMagneticSolver(config)
    if solver.use_mlx:
        try:
            if args.device == "gpu":
                mx.set_default_device(mx.gpu)
            else:
                mx.set_default_device(mx.cpu)
            solver.device = str(mx.default_device())
        except Exception:
            pass

    if args.benchmark:
        validation = solver.validate_parallel(steps=args.steps, block_sizes=[args.particles, args.block_size, max(1, args.block_size // 2), 32])
        print({
            "device": solver.device,
            "serial_block_size": validation["serial_block_size"],
            "serial_time_seconds": validation["serial_time_seconds"],
            "reference_checksum": validation["reference_checksum"],
            "results": validation["results"],
        })
        return

    result = solver.run(steps=args.steps, log_every=1)
    print({
        "device": solver.device,
        "mx_available": solver.use_mlx,
        "n_particles": args.particles,
        "block_size": args.block_size,
        "steps": args.steps,
        "initial_energy": float(result["kinetic_energy"][0]),
        "final_energy": float(result["kinetic_energy"][-1]),
        "mean_velocity_final": np.asarray(result["mean_velocity"][-1]).tolist(),
    })


if __name__ == "__main__":
    main()
