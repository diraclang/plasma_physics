"""Utilities for mode-resolved field reconstruction.

The numerical formulation used throughout this project keeps the x-direction on
an explicit grid and resolves the periodic y,z dependence through a finite set of
Fourier harmonics (k_y, k_z). Each retained mode has a one-dimensional x-array
for the field amplitudes, and the real-space field is reconstructed as

    E(x, y, z) = sum_k E_k(x) * exp(i*(k_y*y + k_z*z))
    B(x, y, z) = sum_k B_k(x) * exp(i*(k_y*y + k_z*z))

where the sum runs over the retained mode bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class FieldMode:
    """One retained Fourier mode with field amplitudes defined on x-grid nodes."""

    ky: float
    kz: float
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray
    x_grid: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.Ex = np.asarray(self.Ex, dtype=np.complex128)
        self.Ey = np.asarray(self.Ey, dtype=np.complex128)
        self.Ez = np.asarray(self.Ez, dtype=np.complex128)
        self.Bx = np.asarray(self.Bx, dtype=np.complex128)
        self.By = np.asarray(self.By, dtype=np.complex128)
        self.Bz = np.asarray(self.Bz, dtype=np.complex128)

        if self.x_grid is None:
            self.x_grid = np.arange(self.Ex.size, dtype=float)
        else:
            self.x_grid = np.asarray(self.x_grid, dtype=float)

        if self.Ex.shape != self.x_grid.shape:
            raise ValueError('FieldMode component arrays must match x_grid length')

    def _interp_complex(self, x: float | np.ndarray, values: np.ndarray):
        x_arr = np.asarray(x, dtype=float)
        if values.dtype.kind == 'c':
            re = np.interp(x_arr, self.x_grid, values.real)
            im = np.interp(x_arr, self.x_grid, values.imag)
            return re + 1j * im
        return np.interp(x_arr, self.x_grid, values)

    def Ex_interp(self, x):
        return self._interp_complex(x, self.Ex)

    def Ey_interp(self, x):
        return self._interp_complex(x, self.Ey)

    def Ez_interp(self, x):
        return self._interp_complex(x, self.Ez)

    def Bx_interp(self, x):
        return self._interp_complex(x, self.Bx)

    def By_interp(self, x):
        return self._interp_complex(x, self.By)

    def Bz_interp(self, x):
        return self._interp_complex(x, self.Bz)


def _mode_phase(ky: float, kz: float, y: float | np.ndarray, z: float | np.ndarray):
    return np.exp(1j * (ky * np.asarray(y) + kz * np.asarray(z)))


def gather_particle_field(xi: float, yi: float, zi: float,
                         modes: Sequence[FieldMode]):
    """Return E = (Ex,Ey,Ez), B = (Bx,By,Bz) at one particle position.

    The total real-space field is reconstructed by summing all retained Fourier
    harmonics at the particle's actual (y,z) phase:

        E(x_i, y_i, z_i) = sum_k E_k(x_i) * exp(i*(k_y*y_i + k_z*z_i))
        B(x_i, y_i, z_i) = sum_k B_k(x_i) * exp(i*(k_y*y_i + k_z*z_i))
    """
    Ex = 0.0j
    Ey = 0.0j
    Ez = 0.0j
    Bx = 0.0j
    By = 0.0j
    Bz = 0.0j

    for mode in modes:
        phase = _mode_phase(mode.ky, mode.kz, yi, zi)
        Ex += mode.Ex_interp(xi) * phase
        Ey += mode.Ey_interp(xi) * phase
        Ez += mode.Ez_interp(xi) * phase
        Bx += mode.Bx_interp(xi) * phase
        By += mode.By_interp(xi) * phase
        Bz += mode.Bz_interp(xi) * phase

    return Ex.real, Ey.real, Ez.real, Bx.real, By.real, Bz.real


def reconstruct_field(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
                      modes: Sequence[FieldMode]):
    """Reconstruct the full 3D field from the retained mode bank.

    Returns a dictionary with the electric and magnetic components as 3D arrays:

        { 'E': (Ex, Ey, Ez), 'B': (Bx, By, Bz) }

    Each component has shape (len(x_grid), len(y_grid), len(z_grid)).
    """
    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)
    zg = np.asarray(z_grid, dtype=float)

    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')

    Ex = np.zeros(X.shape, dtype=np.complex128)
    Ey = np.zeros(X.shape, dtype=np.complex128)
    Ez = np.zeros(X.shape, dtype=np.complex128)
    Bx = np.zeros(X.shape, dtype=np.complex128)
    By = np.zeros(X.shape, dtype=np.complex128)
    Bz = np.zeros(X.shape, dtype=np.complex128)

    for mode in modes:
        phase = np.exp(1j * (mode.ky * Y + mode.kz * Z))
        Ex += np.vectorize(mode.Ex_interp, otypes=[complex])(X) * phase
        Ey += np.vectorize(mode.Ey_interp, otypes=[complex])(X) * phase
        Ez += np.vectorize(mode.Ez_interp, otypes=[complex])(X) * phase
        Bx += np.vectorize(mode.Bx_interp, otypes=[complex])(X) * phase
        By += np.vectorize(mode.By_interp, otypes=[complex])(X) * phase
        Bz += np.vectorize(mode.Bz_interp, otypes=[complex])(X) * phase

    return {
        'E': (Ex.real, Ey.real, Ez.real),
        'B': (Bx.real, By.real, Bz.real),
    }


__all__ = [
    'FieldMode',
    'gather_particle_field',
    'reconstruct_field',
]
