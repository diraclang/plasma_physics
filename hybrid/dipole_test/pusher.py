from __future__ import annotations

import numpy as np


def boris_step(
    x: np.ndarray,
    v: np.ndarray,
    q_over_m: float,
    dt: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one particle by one Boris step."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.asarray(e_field, dtype=float)
    b = np.asarray(b_field, dtype=float)

    half_kick = q_over_m * e * (0.5 * dt)
    v_minus = v + half_kick

    t = q_over_m * b * (0.5 * dt)
    t2 = float(np.dot(t, t))
    s = 2.0 * t / (1.0 + t2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_new = v_plus + half_kick
    x_new = x + v_new * dt
    return x_new, v_new


def boris_step_many(
    x: np.ndarray,
    v: np.ndarray,
    q_over_m: float,
    dt: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance N particles (arrays of shape (N, 3)) by one Boris step."""
    half_kick = q_over_m * e_field * (0.5 * dt)
    v_minus = v + half_kick

    t = q_over_m * b_field * (0.5 * dt)
    t2 = np.einsum("ij,ij->i", t, t)
    s = 2.0 * t / (1.0 + t2)[:, None]

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_new = v_plus + half_kick
    x_new = x + v_new * dt
    return x_new, v_new
