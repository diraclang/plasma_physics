"""Energy bookkeeping for the mode-resolved PIC field solve.

The code stores the transverse electromagnetic fields mode by mode, with the
x-direction on a real grid and the periodic y,z dependence expanded in a
retained set of Fourier modes. The total field energy is therefore obtained by
Parseval's theorem as a sum over the retained mode amplitudes:

    E_field = sum_{modes} [ 0.5 * eps_0 * integral_x |E_mode|^2 dx
                         + 0.5 / mu_0 * integral_x |B_mode|^2 dx ]

This is equivalent to the full real-space box integral, but it makes the
Fourier-mode bookkeeping explicit and is the natural formulation for the mode-
resolved solver used here.

The kinetic energy is the usual PIC sum over particles:

    E_kinetic = sum_j 0.5 * m_j * (v_j^2) * w_j

with w_j the macroparticle weight.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_mode_list(state_or_states):
    if isinstance(state_or_states, (list, tuple)):
        return list(state_or_states)
    return [state_or_states]


def mode_magnetic_energy(state, dx: float, mu_0: float = 1.0) -> float:
    """Return the magnetic energy of one Fourier-mode state.

    The retained fields are already the x-grid amplitudes for a single Fourier
    harmonic, so the total magnetic energy is just the x-integral of |B|^2 in
    that mode. Summing over retained modes is the mode-space analogue of
    integrating |B|^2 over the full box.
    """
    B2 = (
        np.abs(state.Bx) ** 2 +
        np.abs(state.By) ** 2 +
        np.abs(state.Bz) ** 2
    )
    return float(np.real(0.5 / mu_0 * np.sum(B2) * dx))


def mode_electric_energy(state, dx: float, eps_0: float = 0.04) -> float:
    """Return the electric-field energy of one Fourier-mode state."""
    E2 = (
        np.abs(state.ETx) ** 2 +
        np.abs(state.ETy) ** 2 +
        np.abs(state.ETz) ** 2
    )
    return float(np.real(0.5 * eps_0 * np.sum(E2) * dx))


def field_energy(state_or_states, dx: float, eps_0: float = 0.04, mu_0: float = 1.0) -> float:
    """Return the total electric + magnetic field energy for one or more mode states.

    This is the mode-resolved version of the box integral: each retained mode
    contributes its own x-grid energy, and the total is the sum over modes.
    """
    states = _as_mode_list(state_or_states)
    energy = 0.0
    for state in states:
        energy += mode_electric_energy(state, dx, eps_0=eps_0)
        energy += mode_magnetic_energy(state, dx, mu_0=mu_0)
    return float(np.real(energy))


def kinetic_energy(particles) -> float:
    """Compute the kinetic energy for a single ParticleSet or an iterable of them."""
    if particles is None:
        return 0.0

    if isinstance(particles, (list, tuple)):
        return float(np.real(sum(kinetic_energy(p) for p in particles)))

    v2 = particles.vx ** 2 + particles.vy ** 2 + particles.vz ** 2
    return float(np.real(0.5 * particles.m * np.sum(v2) * particles.weight))


def total_energy(state_or_states, dx: float, eps_0: float = 0.04,
                 mu_0: float = 1.0, particles=None) -> float:
    """Return E_total = E_field + E_kinetic.

    The `state_or_states` argument may be either a single TransverseModeState or
    a sequence of them. If `particles` is supplied, their kinetic energy is added.
    """
    E_field = field_energy(state_or_states, dx, eps_0=eps_0, mu_0=mu_0)
    E_kin = kinetic_energy(particles)
    return float(np.real(E_field + E_kin))


__all__ = ['field_energy', 'kinetic_energy', 'total_energy']
