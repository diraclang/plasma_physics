"""Energy bookkeeping for the mode-resolved PIC field solve.

The code stores the transverse electromagnetic fields mode by mode, with the
x-direction on a real grid and the periodic y,z dependence expanded in a
retained set of Fourier modes. The total energy of the system is therefore
computed as

    E_total = E_kinetic + E_electric + E_magnetic

where the field energies are accumulated over the x-grid for every retained
mode:

    E_electric = 0.5 * eps_0 * sum_x (|E_x|^2 + |E_y|^2 + |E_z|^2) dx
    E_magnetic = 0.5 / mu_0 * sum_x (|B_x|^2 + |B_y|^2 + |B_z|^2) dx

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


def field_energy(state_or_states, dx: float, eps_0: float = 0.04, mu_0: float = 1.0) -> float:
    """Return the total electric + magnetic field energy for one or more mode states."""
    states = _as_mode_list(state_or_states)
    energy = 0.0
    for state in states:
        E2 = (
            np.abs(state.ETx) ** 2 +
            np.abs(state.ETy) ** 2 +
            np.abs(state.ETz) ** 2
        )
        B2 = (
            np.abs(state.Bx) ** 2 +
            np.abs(state.By) ** 2 +
            np.abs(state.Bz) ** 2
        )
        energy += 0.5 * eps_0 * np.sum(E2) * dx + 0.5 / mu_0 * np.sum(B2) * dx
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
