import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from config import SimulationConfig, mode_range
from equilibrium import n_eq, derive_units
from particles import load_species


def test_particle_density_matches_equilibrium_in_x_only_normalization():
    cfg = SimulationConfig(
        Lx=64.0,
        Ly=64.0,
        Lz=64.0,
        lam=8.0,
        B0=2.0,
        Bg=2.0,
        n_bg=0.2,
        mass_ratio=4.0,
        c=5.0,
        V_Te=2.0,
        T_i_over_T_e=1.0,
        nx=64,
        particles_per_cell=200,
        dt=0.05,
        modes=mode_range(range(-1, 2), range(-1, 2), seed_amplitude=0.0),
    )
    units = derive_units(cfg)
    rng = np.random.default_rng(1234)
    ion = load_species(cfg, units, 'ion', rng)

    dx = cfg.Lx / cfg.nx
    x = np.arange(cfg.nx) * dx + 0.5 * dx
    counts = np.bincount(np.floor(ion.x / dx).astype(int) % cfg.nx, minlength=cfg.nx).astype(float)
    density = counts * ion.weight / dx
    expected = n_eq(x, cfg)

    # In the reduced x-only formulation, density is measured per unit x-length.
    frac_err = np.max(np.abs(density - expected)) / np.max(expected)
    assert frac_err < 0.5
