# Plasma Physics Project — Handoff Notes

Date: 2026-09-30

## Goal
Reconstruct (with modern tooling, flexible config) a 1993-era 3D-equivalent
electromagnetic PIC code for studying **multiple tearing mode coupling and
magnetic surface breakup** (island overlap / stochastization), using a hybrid
scheme: full kinetic PIC in one gradient direction (x), with the two periodic
"slab" directions (y, z) represented as a small set of retained Fourier
harmonics rather than a grid.

## Physical setup (all confirmed with the user, cross-checked numerically)

**Geometry**: periodic "double Harris sheet" — two mirrored current sheets
placed at x=Lx/4 and x=3Lx/4 inside a periodic box of length Lx, so the whole
domain closes on itself with no special boundary treatment in x.

```
Bz(x) = B0 * [ tanh((x-Lx/4)/lam) - tanh((x-3Lx/4)/lam) - 1 ]   (reversing field)
By(x) = Bg                                                        (uniform guide field)
n(x)  = n0 * [ sech^2((x-Lx/4)/lam) + sech^2((x-3Lx/4)/lam) ] + n_bg
```

**Confirmed 1993 parameters** (all now in `config.SimulationConfig` defaults):
- `Lx=256, Ly=256, Lz=256` (lengths in electron Debye lengths, lambda_De=1)
- `lam=20` (Harris scale length, in lambda_De)
- `B0=4, Bg=4` (reversing field peak and guide field — NOTE: exactly equal!)
- `mass_ratio=4` (m_i/m_e)
- `c=5` (speed of light — genuinely resolved; this is a FULL electromagnetic
  code, not Darwin-approximation; real light-wave CFL constraint applies)
- `V_Te=2` (electron thermal speed; combined with mass_ratio=4 this forces
  `T_i=T_e` exactly — verified consistent with `V_Ti=1`)
- `nx=256` grid cells in x (was 128 single-sheet, doubled for the periodic trick)
- Species: kinetic ions AND electrons (not a fluid-electron hybrid)
- Field solve: **Helmholtz/Coulomb-gauge split** — longitudinal E via Poisson
  equation per Fourier mode (charge-conserving by construction, no need for a
  charge-conserving deposition scheme), transverse (E_T, B) via DIRECT Maxwell
  curl leapfrog per Fourier mode (dB/dt = -curl E_T, dE_T/dt = c^2 curl B -
  J_T/eps0) -- NOT via a vector potential intermediate.
- Modes: user wants FULL FLEXIBILITY to configure arbitrary (m,n) mode sets
  (e.g. m in -3..3, n in -1..1) at run time -- see `config.mode_range()`.
  Two specific modes have been the running example: (m=0,n=1) and (m=1,n=1).

## IMPORTANT CAVEAT: mode (m=1,n=1) does not resonate with these exact numbers
With `Bg=B0` exactly and `Ly=Lz` exactly, mode (1,1) has k_y=k_z, making the
resonance condition `tanh(x_res) = -Bg/B0 = -1` exactly -- a genuine
mathematical degeneracy (resonance pushed to x=+/-infinity, i.e. NO finite
resonant surface for that mode with these exact parameters). This was checked
numerically and is not a bug. If a finite second resonant surface is needed
for mode (1,1) specifically, either `Bg` or `Ly`/`Lz` need to be perturbed
slightly away from exact equality -- this is a config change, not a code
change, since the mode system is already fully flexible.

## Unit system (see `src/units.py` docstring for full derivation)
Fixed base units: `lambda_De=1, mu_0=1, n0=1, m_e=1`. Everything else
(`eps_0, e, omega_pe, omega_pi, d_i, Omega_ci, Omega_ce, v_A`) is DERIVED from
the physics inputs (`mass_ratio, c, B0, V_Te, T_i_over_T_e`). Verified
numerically: `eps_0=0.04, e=0.4, omega_pe=2, omega_pi=1, d_i=5 (lambda_De),
Omega_ci=0.4, Omega_ce=1.6, v_A=2` -- matches `B0/sqrt(mu_0*n0*m_i)=2` exactly
(Omega = eB/m, no 1/c; earlier 0.08/0.32 values were wrong).

## Code status (see TODO list embedded in the session, reproduced here)

DONE and verified:
1. Project structure + config schema (`src/config.py`) -- flexible ModeSpec
   list, `mode_range()` helper for generating mode banks over ranges.
2. Equilibrium profiles (`src/equilibrium.py`) -- Bz_eq, By_eq, n_eq,
   dBz_dx_eq, derive_units, drift_velocities.
3. Equilibrium force balance verified standalone
   (`scripts/check_equilibrium.py`) -- P_total is exactly constant
   analytically (Bz^2 = B0^2 - B0^2*sech^2 identity makes sech^2 terms cancel
   between magnetic and kinetic pressure, regardless of n_bg).
4. Particle initialization (`src/particles.py`) -- inverse-CDF exact sampling
   of x-positions matching n_eq(x) (no rejection-sampling inefficiency),
   sheet/background population split per-particle (probabilistic, based on
   local density mixing fraction), correct thermal speeds and drift
   velocities. VERIFIED via `scripts/check_particles.py`:
   - density profile matches n_eq(x) well (< 3% at sheet center)
   - **BUG FOUND AND FIXED**: `drift_velocities()` in equilibrium.py had the
     WRONG SIGN for v_iy/v_ey (current density came out with flipped sign
     vs. the Ampere's-law expectation -(1/mu_0)dBz/dx). Fixed by negating
     `v_char`. Re-verified: current density now matches in sign and is within
     ~3-17% in magnitude (the remaining error is just particle-count
     statistical noise at particles_per_cell=2000, not a bug).

NOT STARTED (the bulk of remaining work):
5. Boris pusher + multi-mode field gather (interpolate x-grid fields at each
   particle's x, sum over retained Fourier harmonics using the particle's
   actual (y,z) phase, to get the real E,B felt by that particle).
6. Per-mode Poisson solve for longitudinal E (tridiagonal solve in x, one per
   retained (k_y,k_z) harmonic).
7. Per-mode Maxwell curl leapfrog for transverse (E_T, B) (finite-difference
   in x combined with ik_y, ik_z factors for the y,z derivatives).
8. Charge/current deposition projected onto each Fourier mode (accumulate
   each particle's contribution to rho_k(x), J_k(x) weighted by
   e^{-i(k_y*y_particle + k_z*z_particle)}).
9. Main time-stepping loop wiring steps 5-8 together.
10. Diagnostics: island width tracking, Poincare plots of field lines,
    explicit Chirikov overlap parameter, energy conservation check.

## Files
- `src/units.py` -- unit system derivation
- `src/config.py` -- SimulationConfig, ModeSpec, mode_range()
- `src/equilibrium.py` -- equilibrium profiles, derived units, drift velocities
- `src/particles.py` -- ParticleSet, load_species()
- `scripts/check_equilibrium.py` -- standalone equilibrium verification (passing)
- `scripts/check_particles.py` -- standalone particle loading verification (passing)
- `equilibrium_check.png`, `particles_check.png` -- verification plots

## Suggested next steps when resuming
1. Start with the field solver (steps 6-7 above) since the pusher (step 5)
   needs fields to gather from. Recommend building and unit-testing the
   per-mode Poisson solve FIRST in isolation (e.g. verify it reproduces a
   known analytic charge distribution's potential), then the Maxwell leapfrog
   in isolation (e.g. verify a vacuum light wave propagates at the correct
   speed c, and that a static equilibrium current doesn't spuriously radiate).
2. Then build charge/current deposition (step 8), verify it reproduces the
   equilibrium Jy(x) profile when fed the ALREADY-VERIFIED loaded particles
   (cross-check against `scripts/check_particles.py`'s already-passing Jy
   comparison).
3. Only then wire up the full time-stepping loop (step 9) and diagnostics
   (step 10).
4. Given the CFL constraint from real light waves (c=5 here), double check
   the timestep `dt` is small enough: dt < dx/c roughly, where
   dx = Lx/nx = 256/256 = 1 (lambda_De). So dt should be well under 1/5=0.2
   in these units -- current default `dt=0.01` is safely small, but this
   should be explicitly checked once the leapfrog is implemented, since a
   CFL violation would show up as silent numerical instability blowup, not an
   obvious crash.
