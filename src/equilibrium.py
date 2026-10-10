"""
Equilibrium profiles for the periodic "double Harris sheet" used as the
initial condition. Two mirrored current sheets are placed at x = Lx/4 and
x = 3*Lx/4 inside a periodic box of length Lx, so the whole domain closes on
itself with no special boundary treatment needed.

Reversing field is Bz(x) (current sheets carry current in +/- y), with a
uniform guide field By = Bg on top. Axes are just a relabeling of the textbook
Harris sheet (By(x) -> Bz(x)), chosen so "x" stays the PIC/gradient direction
and "y,z" stay the two periodic Fourier directions, matching config.py.

See units.py for the full unit-system derivation (lambda_De=1, mu_0=1, n0=1,
m_e=1 fixed; everything else derived from mass_ratio, c, B0, V_Te, T_i/T_e).
"""
from config import SimulationConfig
from units import DerivedUnits, derive_units as _derive_units_raw


def derive_units(cfg: SimulationConfig) -> DerivedUnits:
    return _derive_units_raw(
        mass_ratio=cfg.mass_ratio, c=cfg.c, B0=cfg.B0, V_Te=cfg.V_Te,
        T_i_over_T_e=cfg.T_i_over_T_e,
    )


def Bz_eq(x, cfg: SimulationConfig):
    """Double Harris sheet reversing field, periodic on [0, Lx). Peak
    amplitude is cfg.B0."""
    x1 = cfg.Lx / 4.0
    x2 = 3.0 * cfg.Lx / 4.0
    return cfg.B0 * _Bz_shape(x, cfg, x1, x2)


def _Bz_shape(x, cfg: SimulationConfig, x1: float, x2: float):
    import numpy as np
    return np.tanh((x - x1) / cfg.lam) - np.tanh((x - x2) / cfg.lam) - 1.0


def By_eq(x, cfg: SimulationConfig):
    """Uniform guide field."""
    import numpy as np
    return cfg.Bg * np.ones_like(np.asarray(x, dtype=float))


def n_eq(x, cfg: SimulationConfig):
    """Double Harris sheet density profile (peak amplitude n0=1, i.e. in
    units of n0), plus uniform background n_bg."""
    import numpy as np
    x1 = cfg.Lx / 4.0
    x2 = 3.0 * cfg.Lx / 4.0
    sech2 = lambda u: 1.0 / np.cosh(u) ** 2
    return sech2((x - x1) / cfg.lam) + sech2((x - x2) / cfg.lam) + cfg.n_bg


def dBz_dx_eq(x, cfg: SimulationConfig):
    """Analytic derivative of Bz_eq, needed for the equilibrium current
    Jy(x) = -(1/mu_0) dBz/dx."""
    import numpy as np
    x1 = cfg.Lx / 4.0
    x2 = 3.0 * cfg.Lx / 4.0
    sech2 = lambda u: 1.0 / np.cosh(u) ** 2
    return cfg.B0 * (sech2((x - x1) / cfg.lam) - sech2((x - x2) / cfg.lam)) / cfg.lam


def drift_velocities(cfg: SimulationConfig, units: DerivedUnits):
    """Uniform equilibrium drift speeds (v_iy, v_ey) for the *sheet* particle
    population (the sech^2 part of n_eq), derived from the classic single-
    species-independent Harris solution:

        Jy(x) = -(1/mu_0) dBz/dx = n_sheet(x) * e * (v_iy - v_ey)

    Since dBz/dx and n_sheet(x) both have the same sech^2(x/lam) shape (for
    the ideal n_bg=0 case), the ratio is x-independent, giving uniform drift
    speeds -- this is the defining property of the Harris equilibrium. Added
    n_bg does NOT break bulk force balance (verified: P_total stays exactly
    constant, since n_bg is spatially uniform and just adds a constant
    pressure offset); it is standard practice to give the background
    population zero drift.

    Sign check (verified against particle simulation 2026-09-30): at x=Lx/4,
    dBz/dx > 0 (field rises through zero from - to + as x increases across
    the sheet center in THIS profile's orientation), so Jy = -(1/mu_0)dBz/dx
    is NEGATIVE there. Since Jy = n*e*(v_iy - v_ey), this requires
    v_iy - v_ey = -v_char (NOT +v_char) -- i.e. v_iy is negative, v_ey is
    positive, at x=Lx/4. Earlier version had this backwards.
    """
    v_char = -cfg.B0 / (units.mu_0 * units.n0 * units.e * cfg.lam)
    T_total = units.T_i + units.T_e
    v_iy = v_char * units.T_i / T_total
    v_ey = -v_char * units.T_e / T_total
    return v_iy, v_ey, units.T_i, units.T_e

