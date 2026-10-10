"""
Simulation configuration: all physics and numerical parameters live here as a
single dataclass, loadable from / saveable to a plain dict (so it can be
serialized to JSON/YAML later without extra dependencies).

Geometry recap:
  - x is the PIC (gradient) direction: the double Harris sheet reversal lives
    here, resolved by real macroparticles + a 1D finite-difference grid.
  - y, z are periodic "slab" directions, represented purely as a small set of
    retained Fourier harmonics (k_y, k_z) -- no grid, no particles-in-y/z.

Equilibrium (in code units, see units.py -- lambda_De=1, mu_0=1, n0=1, m_e=1
are the fixed base units; everything else is derived from the physics inputs
below):
  Bz(x) = B0 * [ tanh((x - Lx/4)/lambda) - tanh((x - 3*Lx/4)/lambda) - 1 ]
  By(x) = Bg                                  (uniform guide field)
  n(x)  = n0 * [ sech^2((x-Lx/4)/lambda) + sech^2((x-3*Lx/4)/lambda) ] + n_bg

This is the "double Harris sheet" trick: two mirrored current sheets placed at
x = Lx/4 and x = 3*Lx/4 inside a periodic box of length Lx, so the whole
domain closes on itself with no special boundary treatment needed in x.

Defaults below reproduce the 1993 run parameters as reconstructed so far:
  lambda_De=1, lam=20, mass_ratio=4, c=5, B0=Bg=4, V_Te=2 (=> V_Ti=1, T_i=T_e),
  nx=256 (128 single-sheet, doubled), Lz=256, Ly=256 (confirmed).

  NOTE: with Bg=B0 exactly and Ly=Lz exactly, any mode with |m|=|n| (e.g. the
  (1,1) mode) has k_y=k_z, making the resonance condition
  tanh(x_res) = -(k_y*Bg)/(k_z*B0) = -Bg/B0 = -1 exactly -- a genuine
  mathematical degeneracy (resonance pushed to x=+/-infinity, i.e. no finite
  resonant surface for that specific mode). This is not a bug; it is a
  property of these exact parameter values. If a finite resonant surface is
  needed for such a mode, either Bg or Ly/Lz must be perturbed slightly away
  from exact equality -- a config change, not a code change.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModeSpec:
    """One retained Fourier harmonic, labeled by integer box mode numbers (m, n).

    k_y = 2*pi*m / Ly
    k_z = 2*pi*n / Lz

    (m, n) = (0, 0) is NOT stored here -- the m=n=0 "mode" is just the
    equilibrium/mean quantities (handled separately, on the x-grid only).
    """
    m: int
    n: int
    # optional small seed perturbation amplitude (fraction of B0) to kick this
    # mode unstable at t=0; if 0.0, only PIC particle noise seeds the mode.
    seed_amplitude: float = 1.0e-4


def mode_range(m_values, n_values, seed_amplitude: float = 1.0e-4,
               exclude_origin: bool = True) -> List[ModeSpec]:
    """Convenience helper: build a list of ModeSpec covering every combination
    of the given m and n integer values. E.g.

        mode_range(m_values=range(-3, 4), n_values=range(-1, 2))

    gives all 7*3=21 combinations of m in {-3,...,3} and n in {-1,0,1} (minus
    the (0,0) origin mode, which is excluded by default since it is not a
    perturbation -- it is just the equilibrium/mean field).
    """
    modes = []
    for m in m_values:
        for n in n_values:
            if exclude_origin and m == 0 and n == 0:
                continue
            modes.append(ModeSpec(m=m, n=n, seed_amplitude=seed_amplitude))
    return modes


@dataclass
class SimulationConfig:
    # --- equilibrium geometry (lengths in lambda_De = 1) ---
    Lx: float = 256.0         # domain length in x (periodic, DOUBLED sheet)
    Ly: float = 256.0         # domain length in y (periodic) -- confirmed
    Lz: float = 256.0         # domain length in z (periodic)
    lam: float = 20.0         # Harris sheet scale length lambda, in lambda_De
    B0: float = 4.0           # reversing field peak amplitude (Bz)
    Bg: float = 4.0           # uniform guide field amplitude (By)
    n_bg: float = 0.2         # uniform background density, in units of n0

    # --- species ---
    mass_ratio: float = 4.0     # m_i / m_e
    c: float = 5.0              # speed of light, in code units (real light waves)
    V_Te: float = 2.0           # electron thermal speed sqrt(T_e/m_e), sets T_e
    T_i_over_T_e: float = 1.0   # ion/electron temperature ratio

    # --- retained Fourier modes (y,z directions) ---
    modes: List[ModeSpec] = field(default_factory=lambda: [
        ModeSpec(m=0, n=1),
        ModeSpec(m=1, n=1),
    ])

    # --- numerics ---
    nx: int = 256                  # number of grid cells in x (post-doubling)
    particles_per_cell: int = 200  # macroparticles per cell per species
    dt: float = 0.05               # upper bound on the timestep, code time units;
                                  # the loop uses coupled_loop.stable_dt(), the
                                  # min of this and the physics limits
                                  # (Omega_ce dt<=0.1, omega_pe dt<=0.2, c dt/dx<=0.5).
    n_steps: int = 20000            # number of timesteps

    # --- diagnostics ---
    diag_every: int = 50           # steps between diagnostic snapshots

    def k_y(self, mode: ModeSpec) -> float:
        import math
        return 2.0 * math.pi * mode.m / self.Ly

    def k_z(self, mode: ModeSpec) -> float:
        import math
        return 2.0 * math.pi * mode.n / self.Lz

    def resonant_x(self, mode: ModeSpec) -> List[float]:
        """Return the x-location(s) in [0, Lx) where k . B(x) = 0 for this mode,
        i.e. k_z * Bz(x) + k_y * Bg = 0, evaluated against the double-Harris-sheet
        equilibrium. Returns an empty list if the mode never resonates
        (e.g. m=n=0, or |k_y*Bg| > k_z*B0 with no solution)."""
        import math
        ky = self.k_y(mode)
        kz = self.k_z(mode)
        if kz == 0.0 and ky == 0.0:
            return []
        x_sheet1 = self.Lx / 4.0
        x_sheet2 = 3.0 * self.Lx / 4.0
        roots = []
        if kz == 0.0:
            # k . B = ky * Bg = const, only resonates if Bg == 0 (degenerate)
            return []
        ratio = -ky * self.Bg / (kz * self.B0)
        if abs(ratio) < 1.0:
            offset = self.lam * math.atanh(ratio)
            roots.append(x_sheet1 + offset)
            roots.append(x_sheet2 - offset)
        return roots
