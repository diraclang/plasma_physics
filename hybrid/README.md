# Hybrid Test-Particle Starter

This directory is a clean starting point for test-particle motion in a dipole field.

## Scope (Stage 1 -> Stage 2)

- Stage 1: Static dipole magnetic field (`E = 0`)
- Stage 2: Spinning dipole with corotation electric field

The current implementation already supports both modes, with a runtime flag to enable corotation.

## Files

- `dipole_test/fields.py`: dipole `B` and corotation `E`
- `dipole_test/pusher.py`: non-relativistic Boris pusher
- `dipole_test/simulate.py`: trajectory integration and diagnostics
- `run_dipole_test.py`: command-line runner
- `run_dipole_ensemble.py`: ensemble generator for density diagnostics
- `plot_dipole_density.py`: azimuth-slice and collapsed-angle density plots
- `plot_sc_meridian_evolution.py`: self-consistent meridian evolution (density + total B field lines)
- `HYBRID_CYLINDRICAL_FORMALISM.md`: self-consistent hybrid equations and cylindrical component form

## Quick Start

Run static dipole test:

```bash
python hybrid/run_dipole_test.py --steps 20000 --dt 0.001
```

Run with corotation electric field:

```bash
python hybrid/run_dipole_test.py --steps 20000 --dt 0.001 --with-corotation --omega 0.05
```

The script writes CSV output to `hybrid/output/trajectory.csv` and prints simple diagnostics.

Run an ensemble (recommended for density maps):

```bash
python hybrid/run_dipole_ensemble.py --n 128 --steps 4000 --dt 0.001
```

Run with axisymmetric self-consistent hybrid electric field (ions kinetic, electrons fluid):

```bash
python hybrid/run_dipole_ensemble.py \
	--n 128 --steps 4000 --dt 0.001 \
	--self-consistent --sc-nr 96 --sc-nz 128 \
	--sc-r-max 10 --sc-z-max 6 \
	--electron-temp 0.05
```

Notes:
- This mode is intentionally conservative by default (Hall OFF, magnetic perturbation evolution OFF) for numerical robustness.
- Enable extra physics gradually with `--with-hall` and `--evolve-magnetic` after baseline checks.

Localized blob release example (near shell `L~5`, no corotation):

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic \
	--n 10000 --steps 600 --dt 0.001 \
	--ring-radius 5.0 --radial-spread 0.1 --z-spread 0.1 \
	--phi-center-deg 0 --phi-spread-deg 8 \
	--speed 0.02 --vz-spread 0.003 \
	--sc-nr 64 --sc-nz 96 --sc-r-max 10 --sc-z-max 6 \
	--e-clip 2 --j-clip 2 --snapshot-stride 20 \
	--output hybrid/output/ensemble_sc_settling_L5_blob_10k.npz
```

Optional outer magnetic-pressure wall (northward field that ramps up outside a radius):

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic \
	--outer-bz 0.05 --outer-bz-r0 8.0 --outer-bz-width 0.8 \
	...
```

This is more useful than a uniform added field when your goal is to keep particles from escaping outward.

If you want a physically cleaner external forcing (e.g., attributed to an external ring/coil current), use a uniform northward bias field:

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic \
	--uniform-bz 0.03 \
	...
```

You can keep `--outer-bz 0` to avoid adding model-dependent non-uniform external structure.

Then render evolution in meridian plane with field lines:

```bash
python hybrid/plot_sc_meridian_evolution.py \
	--input hybrid/output/ensemble_sc_settling_L5_blob_10k.npz \
	--out-prefix hybrid/output/sc_settling_L5_blob_10k \
	--r-max 8 --z-max 4
```

Restart from a previous self-consistent run (continue from last particle and field state):

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic \
	--restart-input hybrid/output/ensemble_sc_settling_L5_blob_10k.npz \
	--steps 600 --dt 0.001 \
	--dipole-scale 20 \
	--sc-nr 64 --sc-nz 96 --sc-r-max 10 --sc-z-max 6 \
	--e-clip 2 --j-clip 2 --snapshot-stride 20 --heartbeat-every 100 \
	--output hybrid/output/ensemble_sc_settling_L5_blob_10k_cont.npz
```

When `--heartbeat-every` is set, the run prints periodic progress lines with step, sim time, density mean, kinetic mean, and elapsed wall time.

## Saved Run Parameters: Dipole-Only 5x Scenario

Fresh run from t=0 to t=12 with dipole ramp 20 -> 100 and no external added fields:

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic --with-hall \
	--hall-density-floor 10 --hall-wdt-max 0.1 \
	--electron-temp 0.0002 --density-floor 1 \
	--n 10000 --steps 12000 --dt 0.001 \
	--ring-radius 5.0 --radial-spread 0.1 --z-spread 0.1 \
	--phi-center-deg 0 --phi-spread-deg 8 \
	--speed 0.02 --vz-spread 0.003 \
	--dipole-scale 20 --dipole-scale-end 100 \
	--uniform-bz 0.0 --outer-bz 0.0 --toroidal-b 0.0 --toroidal-r0 1.0 \
	--sc-nr 64 --sc-nz 96 --sc-r-max 10 --sc-z-max 6 \
	--e-clip 2 --j-clip 2 --snapshot-stride 20 --heartbeat-every 200 \
	--history-stride 50 \
	--output hybrid/output/sc_Te2e4_f1_dipoleonly_ramp5x_fresh_12000.npz
```

Continuation for another 12,000 steps with dipole held at 100 (no further ramp):

```bash
python hybrid/run_dipole_ensemble.py \
	--self-consistent --evolve-magnetic --with-hall \
	--hall-density-floor 10 --hall-wdt-max 0.1 \
	--electron-temp 0.0002 --density-floor 1 \
	--restart-input hybrid/output/sc_Te2e4_f1_dipoleonly_ramp5x_fresh_12000.npz \
	--steps 12000 --dt 0.001 --dipole-scale 100 \
	--uniform-bz 0.0 --outer-bz 0.0 --toroidal-b 0.0 --toroidal-r0 1.0 \
	--sc-nr 64 --sc-nz 96 --sc-r-max 10 --sc-z-max 6 \
	--e-clip 2 --j-clip 2 --snapshot-stride 20 --heartbeat-every 200 \
	--history-stride 50 \
	--output hybrid/output/sc_Te2e4_f1_dipoleonly_hold100_cont_12000.npz
```

With corotation electric field:

```bash
python hybrid/run_dipole_ensemble.py --n 128 --steps 4000 --dt 0.001 --with-corotation --omega 0.05
```

This writes `hybrid/output/ensemble.npz`.

## Density Diagnostics

Create two maps from `ensemble.npz`:

- A narrow azimuth slice around a defined angle
- A collapsed range across azimuth angles (example: `0-30` degrees)

```bash
python hybrid/plot_dipole_density.py \
	--input hybrid/output/ensemble.npz \
	--phi-center 0 --phi-width 10 \
	--phi-min 0 --phi-max 30 \
	--output-prefix hybrid/output/density
```

Outputs:

- `hybrid/output/density_slice.png`
- `hybrid/output/density_collapsed.png`

Both plots are in the meridional `(R, z)` plane and include reference dipole field-line overlays.

## Next Up

- Add adaptive timestep based on local gyrofrequency.
- Add loss boundaries and time-of-flight/loss diagnostics.
- Add phase-space histograms by species and energy band.
- Couple to a field solver for self-consistent hybrid updates.
