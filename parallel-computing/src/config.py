from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpectralConfig:
    Lx: float = 256.0
    Ly: float = 512.0
    Lz: float = 512.0
    dt: float = 0.05
    seed_amplitude: float = 1.0e-3
    x_mode_range: range = field(default_factory=lambda: range(-3, 4))
    y_mode_range: range = field(default_factory=lambda: range(-1, 2))
    z_mode_range: range = field(default_factory=lambda: range(-1, 2))
    include_mean_mode: bool = True

    def mode_count(self) -> int:
        total = 0
        for kx in self.x_mode_range:
            for ky in self.y_mode_range:
                for kz in self.z_mode_range:
                    is_mean = (kx == 0 and ky == 0 and kz == 0)
                    if is_mean and not self.include_mean_mode:
                        continue
                    total += 1
        return total


def build_default_config() -> SpectralConfig:
    return SpectralConfig()
