"""
Particle loading: initialize macroparticles whose statistical moments match
the double Harris sheet equilibrium (equilibrium.py).

Method:
  - x positions are sampled via exact inverse-CDF sampling against n_eq(x)
    (no rejection-sampling inefficiency, and an exact requested particle
    count). This naturally places more particles where the density is
    higher, with uniform per-particle weight -- standard PIC practice.
  - Each particle is independently classified as belonging to the "sheet"
    population (probability n_sheet(x)/n_eq(x) at its own x) or the
    "background" population (probability n_bg/n_eq(x)), since these two
    sub-populations have different velocity distributions (drifting vs not).
  - y, z positions are uniform random on [0,Ly) x [0,Lz) (needed later only
    to evaluate each particle's phase e^{i(k_y y + k_z z)} for the retained
    Fourier harmonics -- there is no y,z grid).
  - Velocities are drawn from a local Maxwellian: vx, vz ~ N(0, vth^2),
    vy ~ N(v_drift, vth^2), with vth and v_drift depending on species and
    sub-population (background has v_drift=0).
"""
from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from units import DerivedUnits
from equilibrium import n_eq, drift_velocities


@dataclass
class ParticleSet:
    species: str       # 'ion' or 'electron'
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    weight: float      # uniform macroparticle weight (same for every particle of this species)
    q: float           # charge of this species (+e or -e)
    m: float           # mass of this species

    @property
    def n(self) -> int:
        return self.x.shape[0]


def _inverse_cdf_sample_x(cfg: SimulationConfig, n_samples: int, rng: np.random.Generator,
                           n_grid: int = 20000, x_max: float | None = None) -> np.ndarray:
    """Exact inverse-CDF sampling of n_samples positions in [0, x_max) (default
    Lx) with PDF proportional to n_eq(x, cfg)."""
    L = cfg.Lx if x_max is None else x_max
    x_grid = np.linspace(0.0, L, n_grid, endpoint=False)
    dens = n_eq(x_grid, cfg)
    cdf = np.concatenate(([0.0], np.cumsum(dens))).astype(float)
    cdf /= cdf[-1]
    u = rng.random(n_samples)
    # interpolate: find x such that cdf(x) = u
    idx = np.interp(u, cdf, np.arange(n_grid + 1))
    # map fractional grid index back to x (grid spacing dx = L/n_grid)
    dx = L / n_grid
    return (idx * dx) % cfg.Lx


def load_species(cfg: SimulationConfig, units: DerivedUnits, species: str,
                  rng: np.random.Generator, half_domain: bool = False) -> ParticleSet:
    """Load one species' macroparticles matching the equilibrium.

    half_domain=True loads only x in [0, Lx/2) (half the particles, same
    weight).  The other half of the double sheet is supplied as the image
    Phi: (x,y,z,vy) -> (x+Lx/2, -y, z, -vy) by coupled_loop (symmetric mode)."""
    if species == 'ion':
        m = units.m_i
        q = units.q_i
    elif species == 'electron':
        m = units.m_e
        q = units.q_e
    else:
        raise ValueError(f"unknown species '{species}'")

    v_iy, v_ey, T_i, T_e = drift_velocities(cfg, units)
    v_drift_sheet = v_iy if species == 'ion' else v_ey
    v_th_sheet = np.sqrt((T_i if species == 'ion' else T_e) / m)

    n_total = round(cfg.particles_per_cell * cfg.nx * _mean_density(cfg))
    if half_domain:
        n_total = round(n_total / 2)
    x = _inverse_cdf_sample_x(cfg, n_total, rng,
                              x_max=cfg.Lx / 2.0 if half_domain else None)

    dens_total = n_eq(x, cfg)
    n_bg_frac = cfg.n_bg / dens_total   # probability this particle is "background"
    is_background = rng.random(n_total) < n_bg_frac

    y = rng.uniform(0.0, cfg.Ly, size=n_total)
    z = rng.uniform(0.0, cfg.Lz, size=n_total)

    vx = rng.normal(0.0, v_th_sheet, size=n_total)
    vz = rng.normal(0.0, v_th_sheet, size=n_total)
    vy = rng.normal(0.0, v_th_sheet, size=n_total)
    # Double Harris sheet: Jy = -dBz/dx changes sign between the two sheets
    # (sheet 1 at Lx/4 occupies x < Lx/2, sheet 2 at 3Lx/4 occupies x >= Lx/2),
    # so the sheet-population drift must reverse in the second sheet.
    sheet_sign = np.where(x < cfg.Lx / 2.0, 1.0, -1.0)
    vy += np.where(is_background, 0.0, v_drift_sheet * sheet_sign)

    weight = _weight(cfg)

    return ParticleSet(species=species, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                        weight=weight, q=q, m=m)


def _mean_density(cfg: SimulationConfig, n_grid: int = 20000) -> float:
    x_grid = np.linspace(0.0, cfg.Lx, n_grid, endpoint=False)
    return float(np.mean(n_eq(x_grid, cfg)))


def _weight(cfg: SimulationConfig) -> float:
    """Physical particles represented per macroparticle, such that
    particles_per_cell macroparticles at the sheet peak density (n=1)
    correspond to a real number density of n0=1 (in code units, since n0=1
    by our unit convention)."""
    cell_volume = (cfg.Lx / cfg.nx) * cfg.Ly * cfg.Lz
    return cell_volume / cfg.particles_per_cell
