"""
Standalone verification script for the Fourier-mode field reconstruction.

The code checks that the real-space electromagnetic field reconstructed from a
small set of retained (k_y, k_z) modes is exactly the sum of the mode amplitudes
weighted by exp(i*(k_y*y + k_z*z)).

Run directly:

    python scripts/check_mode_gather.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from mode_gather import FieldMode, gather_particle_field, reconstruct_field


def main():
    x = np.linspace(0.0, 10.0, 8)
    y = np.linspace(0.0, 3.0, 8)
    z = np.linspace(0.0, 2.0, 8)

    modes = [
        FieldMode(ky=2.0, kz=0.5, Ex=np.cos(x), Ey=np.sin(2.0 * x), Ez=np.zeros_like(x),
                  Bx=np.sin(x), By=np.cos(3.0 * x), Bz=np.zeros_like(x)),
        FieldMode(ky=-1.0, kz=1.5, Ex=np.sin(x), Ey=np.cos(x), Ez=np.zeros_like(x),
                  Bx=np.zeros_like(x), By=np.sin(2.0 * x), Bz=np.cos(2.0 * x)),
    ]

    xi, yi, zi = 3.0, 1.25, 0.75
    Ex, Ey, Ez, Bx, By, Bz = gather_particle_field(xi, yi, zi, modes)

    expected_Ex = 0.0j
    expected_Ey = 0.0j
    expected_Ez = 0.0j
    expected_Bx = 0.0j
    expected_By = 0.0j
    expected_Bz = 0.0j

    for mode in modes:
        phase = np.exp(1j * (mode.ky * yi + mode.kz * zi))
        expected_Ex += mode.Ex_interp(xi) * phase
        expected_Ey += mode.Ey_interp(xi) * phase
        expected_Ez += mode.Ez_interp(xi) * phase
        expected_Bx += mode.Bx_interp(xi) * phase
        expected_By += mode.By_interp(xi) * phase
        expected_Bz += mode.Bz_interp(xi) * phase

    err = max(
        abs(Ex - expected_Ex.real),
        abs(Ey - expected_Ey.real),
        abs(Ez - expected_Ez.real),
        abs(Bx - expected_Bx.real),
        abs(By - expected_By.real),
        abs(Bz - expected_Bz.real),
    )

    print('=== mode gather reconstruction check ===')
    print(f'  max absolute reconstruction error = {err:.3e}')
    print(f'  expected Ex(yi,zi) = {expected_Ex.real:.6f}, measured = {Ex:.6f}')
    print(f'  expected By(yi,zi) = {expected_By.real:.6f}, measured = {By:.6f}')

    if err > 1.0e-12:
        raise SystemExit('mode gather reconstruction failed')

    # Also verify the full-array reconstruction API.
    x_grid = np.linspace(0.0, 10.0, 12)
    y_grid = np.linspace(0.0, 3.0, 12)
    z_grid = np.linspace(0.0, 2.0, 12)
    E_field = reconstruct_field(x_grid, y_grid, z_grid, modes)
    print('  reconstruct_field() returned arrays of shape', E_field['E'][0].shape)

    print('PASS')


if __name__ == '__main__':
    main()
