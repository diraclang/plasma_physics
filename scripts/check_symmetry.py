"""Equivalence test for the symmetric (half-particle + image) mode.

The symmetric run pushes only the real half of the particles and deposits
their images Phi: (x,y,z,vy) -> (x+Lx/2, -y, z, -vy).  Here the same real
particles AND their images are loaded as an explicit full particle set and
pushed in the ordinary (non-symmetric) loop with the same symmetrized seed.
Because Phi is an exact symmetry, both runs must agree to round-off.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

import coupled_loop as cl
import particles
from config import SimulationConfig, mode_range
from particles import ParticleSet


def make_cfg():
    return SimulationConfig(
        Lx=64.0, Ly=64.0, Lz=64.0, lam=10.0, B0=4.0, Bg=4.0, nx=64,
        particles_per_cell=50, dt=0.05,
        modes=mode_range(range(-2, 3), range(-1, 2), seed_amplitude=0.05),
    )


def main():
    cfg = make_cfg()
    n = 40
    h_sym = cl.run_coupled_loop(cfg, n_steps=n, evolve_mean=True, paired_load=True,
                                symmetric=True)

    # explicit full-particle run built from the same half-domain draw
    real = {}
    orig_load = cl.load_species

    def capture(c, units, sp, rng, half_domain=False):
        s = orig_load(c, units, sp, rng, half_domain=True)
        real[sp] = s
        return s

    def full(c, units, sp, rng, half_domain=False):
        r = real[sp]
        if sp == 'electron':
            r.x, r.y, r.z = real['ion'].x.copy(), real['ion'].y.copy(), real['ion'].z.copy()
        im = cl._image(r, c)
        cat = np.concatenate
        return ParticleSet(species=sp, x=cat([r.x, im.x]), y=cat([r.y, im.y]), z=cat([r.z, im.z]),
                           vx=cat([r.vx, im.vx]), vy=cat([r.vy, im.vy]), vz=cat([r.vz, im.vz]),
                           weight=r.weight, q=r.q, m=r.m)

    # fresh (un-advanced) half-domain draw with the same seed, captured with 0 steps
    cl.load_species = capture
    cl.run_coupled_loop(cfg, n_steps=0, evolve_mean=True, paired_load=True, symmetric=True)
    cl.load_species = full

    orig_build = cl.build_modes

    def build_sym(c, include_mean=False):
        m = orig_build(c, include_mean=include_mean)
        cl.symmetrize_fields(m, c.nx)
        return m

    cl.build_modes = build_sym
    h_full = cl.run_coupled_loop(cfg, n_steps=n, evolve_mean=True, paired_load=False,
                                 symmetric=False)
    cl.build_modes, cl.load_species = orig_build, orig_load

    print('=== symmetric (half + image) vs explicit full-particle run ===')
    ok = True
    for key in ('W_K', 'W_E', 'W_B', 'W_B0', 'W_total'):
        a, b = h_sym[key], h_full[key]
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        print(f'  {key:8s} max rel diff = {rel:.2e}')
        ok &= rel < 1.0e-8
    print(f'  field symmetry error after {n} steps = {h_sym["sym_err"]:.2e}')
    ok &= h_sym['sym_err'] < 1.0e-9
    if not ok:
        raise SystemExit('symmetric mode does not reproduce the full-particle run')
    print('PASS')


if __name__ == '__main__':
    main()
