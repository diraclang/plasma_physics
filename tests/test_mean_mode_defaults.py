import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config import SimulationConfig, mode_range
from coupled_loop import build_modes


def test_build_modes_includes_mean_by_default():
    cfg = SimulationConfig(
        Lx=64.0,
        Ly=32.0,
        Lz=32.0,
        nx=16,
        particles_per_cell=2,
        modes=mode_range(range(-1, 2), range(-1, 2), seed_amplitude=0.0),
    )

    modes = build_modes(cfg)
    mean_modes = [md for md in modes if md.is_mean]
    assert len(mean_modes) == 1
    assert mean_modes[0].m == 0 and mean_modes[0].n == 0


def test_build_modes_can_still_disable_mean_mode():
    cfg = SimulationConfig(
        Lx=64.0,
        Ly=32.0,
        Lz=32.0,
        nx=16,
        particles_per_cell=2,
        modes=mode_range(range(-1, 2), range(-1, 2), seed_amplitude=0.0),
    )

    modes = build_modes(cfg, include_mean=False)
    assert all(not md.is_mean for md in modes)
