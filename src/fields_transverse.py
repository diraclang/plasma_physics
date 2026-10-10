"""
Per-mode transverse field solve: direct Maxwell curl leapfrog for (E_T, B),
one independent set of 1D (in x) evolution equations per retained Fourier
harmonic (k_y, k_z):

    dB/dt   = -curl(E_T)
    dE_T/dt = c^2 * curl(B) - J_T / eps_0

In the mixed x-grid / (k_y,k_z)-Fourier representation, the curl of a field F
(with Fourier amplitude components Fx_k(x), Fy_k(x), Fz_k(x)) is:

    (curl F)_x = i*ky*Fz - i*kz*Fy
    (curl F)_y = i*kz*Fx - dFz/dx
    (curl F)_z = dFy/dx  - i*ky*Fx

Time-stepping is a standard leapfrog: B is updated at half-integer time
steps, E_T at integer time steps, using J_T evaluated at the intermediate
half-step (the caller is responsible for supplying J_T at the correct time).

NOTE: J_T must be the TRANSVERSE (divergence-free) part of the deposited
current -- see current_projection.py. Feeding in the full current here would
double-count the longitudinal physics already handled by the separate
Poisson solve for the scalar potential (fields_longitudinal.py).

Grid convention: collocated (not Yee-staggered) -- all field components live
on the same x grid points, x-derivatives via 2nd-order central differences,
consistent with the Poisson solver's grid convention.
"""
from dataclasses import dataclass

import numpy as np


def _central_diff_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)


@dataclass
class TransverseModeState:
    """Field state for one retained Fourier harmonic. B lags E_T by half a
    timestep (leapfrog convention)."""
    ky: float
    kz: float
    ETx: np.ndarray
    ETy: np.ndarray
    ETz: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray


def curl_mode(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray,
               ky: float, kz: float, dx: float):
    dFz_dx = _central_diff_periodic(Fz, dx)
    dFy_dx = _central_diff_periodic(Fy, dx)
    curl_x = 1j * ky * Fz - 1j * kz * Fy
    curl_y = 1j * kz * Fx - dFz_dx
    curl_z = dFy_dx - 1j * ky * Fx
    return curl_x, curl_y, curl_z


def leapfrog_step(state: TransverseModeState, JTx: np.ndarray, JTy: np.ndarray,
                   JTz: np.ndarray, dx: float, dt: float, c: float, eps_0: float) -> None:
    """Advance B by one half-step and E_T by one full step, in place.
    JT* should be the transverse current evaluated at the intermediate
    (post-B-update) half-time-step."""
    curlE_x, curlE_y, curlE_z = curl_mode(state.ETx, state.ETy, state.ETz,
                                          state.ky, state.kz, dx)
    state.Bx = state.Bx - dt * curlE_x
    state.By = state.By - dt * curlE_y
    state.Bz = state.Bz - dt * curlE_z

    curlB_x, curlB_y, curlB_z = curl_mode(state.Bx, state.By, state.Bz,
                                          state.ky, state.kz, dx)
    c2 = c * c
    state.ETx = state.ETx + dt * (c2 * curlB_x - JTx / eps_0)
    state.ETy = state.ETy + dt * (c2 * curlB_y - JTy / eps_0)
    state.ETz = state.ETz + dt * (c2 * curlB_z - JTz / eps_0)
