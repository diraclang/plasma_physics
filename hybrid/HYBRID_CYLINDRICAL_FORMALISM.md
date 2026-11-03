# Hybrid Formalism in Cylindrical Coordinates

This note describes a practical hybrid model for magnetized plasma in cylindrical coordinates.

Target model:
- Ions are particles (kinetic).
- Electrons are a fluid (not particles).
- Fields are solved self-consistently on a grid.

This is written to be implementation-oriented, using simple component formulas.

## 1) Coordinates and Components

Use cylindrical coordinates:
- `r`: radial distance from axis
- `phi`: azimuth angle around axis
- `z`: vertical direction

For any vector `A`:
- `A = (Ar, Aphi, Az)`

## 2) Hybrid Core Equations

### 2.1 Ion equations (particles)

For each ion macro-particle:
- `dx/dt = v`
- `dv/dt = (qi/mi) * (E + v x B)`

Use Boris pusher for time stepping.

### 2.2 Electron closure (massless fluid)

Assume quasi-neutrality:
- `ne ~= ni = n`

Current from magnetic field:
- `J = (1/mu0) * curl(B)`

Electron bulk velocity:
- `ue = ui - J/(e*n)`

Ohm law used in hybrid model:
- `E = - ui x B + (J x B)/(e*n) - grad(pe)/(e*n) + eta*J`

Common closure choices:
- Isothermal electrons: `pe = n * kB * Te`
- Polytropic electrons: `pe = C * n^gamma`

### 2.3 Magnetic field evolution

Faraday law:
- `dB/dt = - curl(E)`

## 3) Curl in Cylindrical Coordinates

For `A = (Ar, Aphi, Az)`:

- `curl(A)_r   = (1/r) * d(Az)/d(phi) - d(Aphi)/d(z)`
- `curl(A)_phi = d(Ar)/d(z) - d(Az)/d(r)`
- `curl(A)_z   = (1/r) * ( d(r*Aphi)/d(r) - d(Ar)/d(phi) )`

So:
- `dBr/dt   = - curl(E)_r`
- `dBphi/dt = - curl(E)_phi`
- `dBz/dt   = - curl(E)_z`

And:
- `Jr   = (1/mu0) * curl(B)_r`
- `Jphi = (1/mu0) * curl(B)_phi`
- `Jz   = (1/mu0) * curl(B)_z`

## 4) Mode Expansion in Azimuth (Cost Reduction)

If azimuthal variation is smooth, expand fields in `phi` modes:
- `Q(r,phi,z,t) = sum_m Qhat_m(r,z,t) * exp(i*m*phi)`

Key simplification:
- `d()/d(phi) -> i*m*( )`

That means each mode is solved on a 2D meridional grid `(r,z)`.

Practical choice:
- keep a few modes only, for example `m = 0,1,2,3`

## 5) Time Step Loop (Hybrid PIC)

At each step:
1. Deposit ion moments to grid: `n`, `ui` (or `Ji`).
2. Compute `J = curl(B)/mu0` on grid.
3. Compute electron pressure term `grad(pe)` from closure.
4. Build `E` from Ohm law.
5. Update `B` with Faraday law.
6. Gather `E` and `B` from grid to particle positions.
7. Push ions with Boris.
8. Apply boundaries and optional filtering.

For mode-expanded version:
1. Deposit modal moments (`m = 0..M`).
2. Solve per-mode field updates on `(r,z)`.
3. Reconstruct fields at particle `phi` during gather.

## 6) Field Splitting for Strong Dipole Background

For dipole systems, split total fields into background + perturbation:
- `B = B0 + dB`
- `E = Eext + dE`

Where:
- `B0`: analytic dipole background
- `Eext`: optional external corotation term
- solve only `dB`, `dE` on grid

Why this helps:
- better numerical conditioning
- less cancellation error in strong-background regions

## 7) Boundary and Axis Handling

### 7.1 Axis (`r = 0`)

Cylindrical operators include `1/r`. Near axis, enforce regularity:
- use parity rules for components
- avoid direct division by zero
- use axis-specific stencils

### 7.2 Physical boundaries

Define clearly:
- inner boundary (planet loss/absorption)
- outer radial and top/bottom boundaries (open, reflecting, or damping)
- mode boundary treatment consistent for all `m`

## 8) Stability and Accuracy Notes

- Use charge-conserving deposition if evolving full EM currents from particles.
- Keep `div(B)` controlled (constrained transport or cleaning).
- Start with small `eta` for robustness.
- Verify convergence in:
  - grid spacing
  - time step
  - number of particles per cell
  - number of retained azimuthal modes

## 9) Diagnostics to Track Every Run

Minimum diagnostics:
- total ion kinetic energy
- magnetic energy of perturbation field
- mode energy spectrum vs time
- `div(B)` norm
- shell drift metrics (`L` or equivalent)
- `mu` drift statistics where adiabatic conditions apply

If these are not stable or converged, do not trust physics conclusions yet.

## 10) Minimal Implementation Path

Recommended order:
1. Axisymmetric hybrid (`m = 0` only).
2. Add one non-axisymmetric mode (`m = 1`).
3. Add `m = 2,3` only if mode-energy diagnostics show need.
4. Compare against full-phi reference for a short benchmark run.

This gives a practical path to self-consistent hybrid physics with much lower cost than full 3D grids.
