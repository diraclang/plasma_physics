# Simulation -> SI Mapping (Dipole-Only Runs)

This page gives a direct mapping from the current normalized dipole run to SI units.

## 1) Normalized anchor from this run

For the dipole-only configuration used in recent runs:

- dipole moment: `m* = 100`
- blob radius: `r* = 5`
- equatorial dipole field at blob:

`|B*_blob| = m* / (r*^3) = 100 / 125 = 0.8`

So if you choose a real field target at blob location, `B_blob [T]`, then:

`B0 = B_blob / 0.8`

Examples:

- `B_blob = 2 T` -> `B0 = 2.5 T`
- `B_blob = 5 T` -> `B0 = 6.25 T`
- `B_blob = 10 T` -> `B0 = 12.5 T`

## 2) Dipole moment required in SI

Dipole equatorial field relation:

`B_eq = (mu0 / (4*pi)) * (m / R^3)`

`m = (4*pi / mu0) * B_eq * R^3 = 1e7 * B_eq[T] * R^3[m^3]  (A*m^2)`

Quick values:

| Blob radius R (m) | B_eq = 2 T | B_eq = 5 T | B_eq = 10 T |
|---:|---:|---:|---:|
| 0.5 | 2.50e6 | 6.25e6 | 1.25e7 |
| 1.0 | 2.00e7 | 5.00e7 | 1.00e8 |
| 1.5 | 6.75e7 | 1.69e8 | 3.38e8 |
| 2.0 | 1.60e8 | 4.00e8 | 8.00e8 |

All moments above are in `A·m^2`.

## 3) Density from beta and ion temperature

Use pressure balance:

`beta = 2*mu0*p / B^2`, with `p ~= n*(Ti + Te)*e`

`n = beta*B^2 / (2*mu0*(Ti + Te)*e)`

`T_i`, `T_e` are in eV. If `T_i` is in MeV: `T_i[eV] = Ti[MeV] × 1e6`.

### Table (Te << Ti)

`n` in `m^-3`.

| Ti (MeV) | B (T) | beta=0.01 | beta=0.05 | beta=0.1 |
|---:|---:|---:|---:|---:|
| 0.01 | 2  | 9.93e18 | 4.97e19 | 9.93e19 |
| 0.01 | 5  | 6.21e19 | 3.10e20 | 6.21e20 |
| 0.01 | 10 | 2.48e20 | 1.24e21 | 2.48e21 |
| 0.1  | 2  | 9.93e17 | 4.97e18 | 9.93e18 |
| 0.1  | 5  | 6.21e18 | 3.10e19 | 6.21e19 |
| 0.1  | 10 | 2.48e19 | 1.24e20 | 2.48e20 |
| 1.0  | 2  | 9.93e16 | 4.97e17 | 9.93e17 |
| 1.0  | 5  | 6.21e17 | 3.10e18 | 6.21e18 |
| 1.0  | 10 | 2.48e18 | 1.24e19 | 2.48e19 |

If `Te ~ Ti`, divide the table density by 2.

## 4) Practical reading for this project

- For fusion-relevant thermal ions (`Ti ~ 10-20 keV = 0.01-0.02 MeV`) and `B_blob ~ 5 T`, density around `1e20 m^-3` is plausible for `beta` in a few percent to ten percent.
- A strong dipole can be physically meaningful, but full reactor realism still needs transport, losses, radiation, and engineering constraints beyond the present normalized model.

## 5) Concrete design point (baseline for next simulations)

### Design Point A

- Ion species: Deuterium ions (D+)
- Blob radius target: `R_blob = 1.0 m`
- Field target at blob center: `B_eq_blob = 5 T`
- Temperature target: `Ti = 10 keV`, `Te = 10 keV`
- Beta target: `beta = 0.05`

### Physical numbers implied

1. Dipole moment requirement

- Formula: `m = 1e7 * B_eq[T] * R^3[m^3]` (A*m^2)
- Result: `m = 1e7 * 5 * 1^3 = 5.0e7 A*m^2`

2. Density requirement from beta

- Formula: `n = beta * B^2 / (2*mu0*(Ti+Te)*e)`
- Here `Ti+Te = 20 keV = 2.0e4 eV`
- Result: `n = 1.55e20 m^-3`

### Mapping to current normalized run

Current normalized anchor from this run:

- `m* = 100`, `r*_blob = 5`, `|B*_blob| = 0.8`

Mapping for Design Point A:

1. Field scale

- `B0 = B_blob / 0.8 = 5 / 0.8 = 6.25 T`

2. Length scale

- `L0 = R_blob / r*_blob = 1.0 / 5 = 0.2 m`

3. Domain size in SI

- `r*_max = 10  -> r_max = 2.0 m`
- `z*_max = 6   -> z_max = 1.2 m`

4. Time and speed scale (D+, using B0 = 6.25 T)

- `t0 = m_i / (q_i*B0) = 3.34 ns`
- `v0 = L0 / t0 = 6.0e7 m/s`
- If `v* = 0.02`, then `v = 1.2e6 m/s` (about `15 keV` for D+)

### Why this point is useful

- `B_eq = 5 T` is demanding but much more practical than `10 T`.
- `m = 5e7 A*m^2` is large but within large superconducting-magnet design space.
- `n ~ 1e20 m^-3` and `Ti ~ 10 keV` are in a fusion-relevant order of magnitude.
