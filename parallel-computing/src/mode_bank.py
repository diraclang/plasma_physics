from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None


@dataclass(frozen=True)
class ModeSpec:
    kx: int
    ky: int
    kz: int
    is_mean: bool = False
    weight: float = 1.0


class ModeBank:
    """Retained Fourier mode bank for a direct spectral 3D solver."""

    def __init__(
        self,
        kx_range: range = range(-3, 4),
        ky_range: range = range(-1, 2),
        kz_range: range = range(-1, 2),
        include_mean: bool = True,
    ) -> None:
        self.kx_range = kx_range
        self.ky_range = ky_range
        self.kz_range = kz_range
        self.include_mean = include_mean
        self.modes = self._build_modes()
        self.kx = np.array([m.kx for m in self.modes], dtype=np.int32)
        self.ky = np.array([m.ky for m in self.modes], dtype=np.int32)
        self.kz = np.array([m.kz for m in self.modes], dtype=np.int32)
        self.weights = np.array([m.weight for m in self.modes], dtype=np.float64)

    def _build_modes(self) -> list[ModeSpec]:
        modes: list[ModeSpec] = []
        for kx in self.kx_range:
            for ky in self.ky_range:
                for kz in self.kz_range:
                    is_mean = (kx == 0 and ky == 0 and kz == 0)
                    if is_mean and not self.include_mean:
                        continue
                    modes.append(
                        ModeSpec(
                            kx=kx,
                            ky=ky,
                            kz=kz,
                            is_mean=is_mean,
                            weight=1.0,
                        )
                    )
        return modes

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)

    def phase_factors(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return exp(i * (kx*x + ky*y + kz*z)) for each retained mode and each point.

        The returned shape is (n_modes, n_particles).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        kx = self.kx[:, None]
        ky = self.ky[:, None]
        kz = self.kz[:, None]
        phase = np.exp(1j * (kx * x[None, :] + ky * y[None, :] + kz * z[None, :]))

        if mx is not None:
            return mx.array(phase)
        return phase

    def summary(self) -> dict[str, int]:
        return {
            "n_modes": len(self.modes),
            "kx_range": len(self.kx_range),
            "ky_range": len(self.ky_range),
            "kz_range": len(self.kz_range),
        }
