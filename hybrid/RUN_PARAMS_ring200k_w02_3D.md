# ring200k_w02_3D simulation parameters

This file records the run settings that are directly verifiable from the saved NPZ outputs.

## Dataset chain

- hybrid/output/ring200k_w02_3D_ramp.npz: t = 0.0 -> 12.0
- hybrid/output/ring200k_w02_3D_c2.npz: t = 12.0 -> 24.0
- hybrid/output/ring200k_w02_3D_c3.npz: t = 24.0 -> 36.0
- hybrid/output/ring200k_w02_3D_c3_plus12k.npz: t = 36.0 -> 48.0
- hybrid/output/ring200k_w02_3D_c3_plus24k.npz: t = 48.0 -> 60.0
- hybrid/output/ring200k_w02_3D_c3_plus36k.npz: t = 60.0 -> 72.0

Total simulated time to final dataset: t = 72.0

## Final dataset metadata (ring200k_w02_3D_c3_plus36k.npz)

- nphi: 16
- with_corotation: false
- omega: 0.05
- self_consistent: true
- dipole_mz_t: constant 100.0 over this segment
- uniform_bz_t: constant 0.0 over this segment

## Saved array dimensions (final dataset)

- x: (200000, 241, 3)
- v: (200000, 241, 3)
- K: (200000, 241)
- mu: (200000, 241)
- t: (241,)
- hist_step: (241,)
- r_grid: (64,)
- z_grid: (96,)
- snap_n: (601, 64, 96)
- snap_dbr: (601, 64, 96)
- snap_dbphi: (601, 64, 96)
- snap_dbz: (601, 64, 96)
- mode_n_t: (12001, 9)
- mode_db_t: (12001, 9)
- snap3d_n: (61, 64, 16, 96)
- snap3d_dbr: (61, 64, 16, 96)
- snap3d_dbphi: (61, 64, 16, 96)
- snap3d_dbz: (61, 64, 16, 96)

## Time-step interpretation

For this continuation segment:

- segment start time: 60.0
- segment end time: 72.0
- segment internal steps: 12000
- implied dt: (72.0 - 60.0) / 12000 = 0.001

Therefore, the final absolute simulation step in the chain is approximately 72000.
