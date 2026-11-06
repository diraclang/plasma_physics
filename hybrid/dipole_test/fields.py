from __future__ import annotations

import numpy as np


def uniform_northward_b_field(bz_uniform: float) -> np.ndarray:
    """Return a spatially uniform northward (+z) magnetic bias field."""
    if bz_uniform == 0.0:
        return np.zeros(3, dtype=float)
    return np.array([0.0, 0.0, float(bz_uniform)], dtype=float)


def dipole_b_field(
    position: np.ndarray,
    dipole_moment: np.ndarray,
    softening_radius: float = 1e-3,
) -> np.ndarray:
    """Return dipole magnetic field at `position` in normalized units.

    Uses: B = 3 r (m·r) / r^5 - m / r^3.
    """
    r = np.asarray(position, dtype=float)
    m = np.asarray(dipole_moment, dtype=float)
    r2 = float(np.dot(r, r))
    rmag = max(np.sqrt(r2), softening_radius)
    mdotr = float(np.dot(m, r))
    return 3.0 * r * mdotr / (rmag**5) - m / (rmag**3)


def corotation_e_field(
    position: np.ndarray,
    magnetic_field: np.ndarray,
    omega_vec: np.ndarray,
) -> np.ndarray:
    """Return corotation electric field E = -(Omega x r) x B."""
    r = np.asarray(position, dtype=float)
    b = np.asarray(magnetic_field, dtype=float)
    omega = np.asarray(omega_vec, dtype=float)
    return -np.cross(np.cross(omega, r), b)


def northward_confining_b_field(
    position: np.ndarray,
    bz_outer: float,
    ramp_center_r: float,
    ramp_width: float,
) -> np.ndarray:
    """Return an axisymmetric +z field that ramps up outside a radius.

    This creates a magnetic-pressure wall on the outer region rather than
    adding a uniform background everywhere.
    """
    if bz_outer == 0.0:
        return np.zeros(3, dtype=float)

    r = np.asarray(position, dtype=float)
    rr = np.hypot(r[0], r[1])
    width = max(float(ramp_width), 1e-6)
    xi = (rr - float(ramp_center_r)) / width
    ramp = 0.5 * (1.0 + np.tanh(xi))
    return np.array([0.0, 0.0, float(bz_outer) * ramp], dtype=float)


def toroidal_b_field(
    position: np.ndarray,
    bphi_strength: float,
    core_radius: float = 1.0,
) -> np.ndarray:
    """Return a background toroidal field B = B_phi e_phi from a central axial current.

    A current along the z-axis produces an azimuthal field.  We approximate the
    profile with a simple core/outer model:

        B_phi = B0 * (r / r_core)           for r < r_core
        B_phi = B0 * (r_core / r)          for r >= r_core

    This keeps the field finite at the axis and captures the expected long-wire
    falloff away from the core.
    """
    if bphi_strength == 0.0:
        return np.zeros(3, dtype=float)

    pos = np.asarray(position, dtype=float)
    rxy = np.hypot(pos[0], pos[1])
    if rxy == 0.0:
        return np.zeros(3, dtype=float)

    r_core = max(float(core_radius), 1e-6)
    if rxy < r_core:
        bphi = float(bphi_strength) * (rxy / r_core)
    else:
        bphi = float(bphi_strength) * (r_core / rxy)

    phi = np.arctan2(pos[1], pos[0])
    return np.array([
        -bphi * np.sin(phi),
        bphi * np.cos(phi),
        0.0,
    ], dtype=float)
