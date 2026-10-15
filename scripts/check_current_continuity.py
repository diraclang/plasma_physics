"""Check the corrected current continuity in the mode-resolved PIC loop.

The repaired current deposition uses a continuity-based Fourier-space fix for the
x-component; the residual should remain small when the density change and the
current divergence are evaluated at the same time level.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from config import SimulationConfig, ModeSpec
from coupled_loop import _cic, _ddx, _phase, build_modes, deposit_rho
from particles import load_species
from equilibrium import derive_units as derive_cfg_units
from numpy.random import default_rng


def main():
    cfg = SimulationConfig(
        Lx=128.0,
        Ly=128.0,
        Lz=128.0,
        lam=20.0,
        B0=4.0,
        Bg=4.0,
        n_bg=0.2,
        mass_ratio=4.0,
        c=5.0,
        V_Te=2.0,
        T_i_over_T_e=1.0,
        modes=[ModeSpec(m=0, n=1), ModeSpec(m=1, n=1)],
        nx=64,
        particles_per_cell=16,
        dt=0.05,
    )
    rng = default_rng(0)
    units = derive_cfg_units(cfg)
    species = [load_species(cfg, units, 'ion', rng, half_domain=False),
               load_species(cfg, units, 'electron', rng, half_domain=False)]
    modes = build_modes(cfg, include_mean=False)
    dx = cfg.Lx / cfg.nx

    # Take a small density perturbation by shifting particles a little in x.
    rho_before = deposit_rho(species, cfg, modes, dx,
                             [_cic(sp.x, dx, cfg.nx) for sp in species],
                             [[_phase(md, sp.y, sp.z) for md in modes] for sp in species])
    for sp in species:
        sp.x = np.mod(sp.x + 0.35 * dx, cfg.Lx)
    shapes_after = [_cic(sp.x, dx, cfg.nx) for sp in species]
    phases_after = [[_phase(md, sp.y, sp.z) for md in modes] for sp in species]
    rho_after = deposit_rho(species, cfg, modes, dx, shapes_after, phases_after)

    mid_states = []
    for sp, (i0, i1, frac) in zip(species, shapes_after):
        mid_states.append(SimpleNamespace(
            x=sp.x,
            y=sp.y,
            z=sp.z,
            vx=sp.vx,
            vy=sp.vy,
            vz=sp.vz,
            q=sp.q,
            weight=sp.weight,
        ))
    J = deposit_current_with_continuity(mid_states, cfg, modes, dx,
                                       shapes_after, phases_after,
                                       rho_before, rho_after, cfg.dt)

    max_defect = 0.0
    for km, md in enumerate(modes):
        divJ = _ddx(J[km][0], dx) + 1j * md.ky * J[km][1] + 1j * md.kz * J[km][2]
        defect = (rho_after[km] - rho_before[km]) / cfg.dt + divJ
        max_defect = max(max_defect, float(np.max(np.abs(defect))))

    print(f'max continuity defect with corrected current = {max_defect:.6e}')
    print('This diagnostic reflects the remaining discrete continuity error in the current projection; it does not yet enforce a zero residual.')


if __name__ == '__main__':
    main()
