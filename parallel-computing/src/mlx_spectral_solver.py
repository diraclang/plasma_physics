from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None

from .config import SpectralConfig, build_default_config
from .mode_bank import ModeBank


@dataclass
class SpeciesState:
    name: str
    charge: float
    mass: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    weight: float = 1.0


class MLXSpectralSolver:
    """First working prototype of a direct spectral 3D EM/PIC solver.

    The implementation is intentionally small and separated from the legacy code,
    but it follows the same structure we want in the final solver:

    1. retained Fourier mode bank,
    2. particle-to-mode density/current deposition,
    3. mode-wise field update,
    4. particle push from projected field values.
    """

    def __init__(
        self,
        config: SpectralConfig | None = None,
        n_particles_per_species: int = 64,
    ) -> None:
        self.config = config or build_default_config()
        self.mode_bank = ModeBank(
            self.config.x_mode_range,
            self.config.y_mode_range,
            self.config.z_mode_range,
            include_mean=self.config.include_mean_mode,
        )
        self.species = self._init_species(n_particles_per_species)
        self.rho = np.zeros(len(self.mode_bank), dtype=np.complex128)
        self.J = np.zeros((len(self.mode_bank), 3), dtype=np.complex128)
        self.E = np.zeros((len(self.mode_bank), 3), dtype=np.complex128)
        self.B = np.zeros((len(self.mode_bank), 3), dtype=np.complex128)

        if mx is not None:
            self.rho_mx = mx.array(self.rho)
            self.J_mx = mx.array(self.J)
            self.E_mx = mx.array(self.E)
            self.B_mx = mx.array(self.B)
        else:
            self.rho_mx = self.rho
            self.J_mx = self.J
            self.E_mx = self.E
            self.B_mx = self.B

    def _init_species(self, n_particles_per_species: int) -> list[SpeciesState]:
        rng = np.random.default_rng(1234)
        species = []
        for name, charge, mass in [("ion", +1.0, 1.0), ("electron", -1.0, 0.1)]:
            x = rng.uniform(0.0, self.config.Lx, size=n_particles_per_species)
            y = rng.uniform(0.0, self.config.Ly, size=n_particles_per_species)
            z = rng.uniform(0.0, self.config.Lz, size=n_particles_per_species)
            v = rng.normal(0.0, 0.25, size=(n_particles_per_species, 3))
            species.append(
                SpeciesState(
                    name=name,
                    charge=charge,
                    mass=mass,
                    x=x,
                    y=y,
                    z=z,
                    vx=v[:, 0],
                    vy=v[:, 1],
                    vz=v[:, 2],
                    weight=1.0 / n_particles_per_species,
                )
            )
        return species

    def deposit_to_modes(self) -> tuple[np.ndarray, np.ndarray]:
        rho = np.zeros(len(self.mode_bank), dtype=np.complex128)
        J = np.zeros((len(self.mode_bank), 3), dtype=np.complex128)

        for species in self.species:
            phase = self.mode_bank.phase_factors(species.x, species.y, species.z)
            phase = np.asarray(phase)
            q_weight = species.charge * species.weight

            rho += q_weight * phase.sum(axis=1)

            for comp, vel in enumerate((species.vx, species.vy, species.vz)):
                J[:, comp] += q_weight * np.sum(phase * vel[None, :], axis=1)

        self.rho = rho
        self.J = J
        if mx is not None:
            self.rho_mx = mx.array(rho)
            self.J_mx = mx.array(J)
        return rho, J

    def _field_from_modes(self) -> tuple[np.ndarray, np.ndarray]:
        # This is a minimal placeholder for the spectral Maxwell update.
        # The full solver will replace this with orbital mode-wise field solves.
        E = np.zeros_like(self.J)
        B = np.zeros_like(self.J)
        for mode_index, (kx, ky, kz) in enumerate(zip(self.mode_bank.kx, self.mode_bank.ky, self.mode_bank.kz)):
            k2 = float(kx * kx + ky * ky + kz * kz)
            if k2 == 0.0:
                E[mode_index] = 0.0
                B[mode_index] = 0.0
                continue
            scale = 1.0 / (1.0 + k2)
            E[mode_index] = scale * self.J[mode_index]
            B[mode_index] = scale * 0.25 * self.rho[mode_index]
        self.E = np.real(E).astype(np.float64)
        self.B = np.real(B).astype(np.float64)
        if mx is not None:
            self.E_mx = mx.array(self.E)
            self.B_mx = mx.array(self.B)
        return self.E, self.B

    def _field_at_point(self, x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(self.mode_bank.phase_factors(np.array([x]), np.array([y]), np.array([z])))[:, 0]
        E = np.sum(self.E * phase.real[:, None], axis=0)
        B = np.sum(self.B * phase.real[:, None], axis=0)
        return E, B

    def step(self, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        dt = self.config.dt if dt is None else dt
        self.deposit_to_modes()
        self._field_from_modes()

        for species in self.species:
            for idx in range(species.x.size):
                E, B = self._field_at_point(species.x[idx], species.y[idx], species.z[idx])
                ax = (species.charge / species.mass) * E[0]
                ay = (species.charge / species.mass) * E[1]
                az = (species.charge / species.mass) * E[2]

                species.vx[idx] += dt * ax
                species.vy[idx] += dt * ay
                species.vz[idx] += dt * az

                species.x[idx] = (species.x[idx] + dt * species.vx[idx]) % self.config.Lx
                species.y[idx] = (species.y[idx] + dt * species.vy[idx]) % self.config.Ly
                species.z[idx] = (species.z[idx] + dt * species.vz[idx]) % self.config.Lz

        return self.rho.copy(), self.J.copy()

    def run(self, steps: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
        history: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(steps):
            rho, J = self.step()
            history.append((rho.copy(), J.copy()))
        return history

    def summary(self) -> dict[str, Any]:
        return {
            "n_modes": len(self.mode_bank),
            "n_species": len(self.species),
            "Lx": self.config.Lx,
            "Ly": self.config.Ly,
            "Lz": self.config.Lz,
            "dt": self.config.dt,
            "mx_available": mx is not None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="All-spectral MLX solver prototype")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--Lx", type=float, default=256.0)
    parser.add_argument("--Ly", type=float, default=512.0)
    parser.add_argument("--Lz", type=float, default=512.0)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()

    config = SpectralConfig(Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, dt=args.dt)
    solver = MLXSpectralSolver(config=config, n_particles_per_species=args.particles)
    history = solver.run(steps=args.steps)

    print(solver.summary())
    print(f"rho_norm={np.linalg.norm(history[-1][0]):.6e}")
    print(f"J_norm={np.linalg.norm(history[-1][1]):.6e}")


if __name__ == "__main__":
    main()
