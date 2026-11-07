from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dipole_test.simulate import (
    DipoleRunConfig,
    EnsembleConfig,
    SelfConsistentRestartState,
    SelfConsistentConfig,
    run_particle_ensemble,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an ensemble of dipole test particles.")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--q-over-m", type=float, default=1.0)
    p.add_argument("--dipole-scale", type=float, default=1.0, help="Scale factor for dipole moment magnitude")
    p.add_argument("--with-corotation", action="store_true")
    p.add_argument("--omega", type=float, default=0.05)
    p.add_argument("--uniform-bz", type=float, default=0.0, help="Uniform external northward (+z) bias field")
    p.add_argument("--outer-bz", type=float, default=0.0, help="Northward confining field strength outside ramp")
    p.add_argument("--outer-bz-r0", type=float, default=8.0, help="Ramp center radius for outer confining field")
    p.add_argument("--outer-bz-width", type=float, default=0.8, help="Ramp width for outer confining field")
    p.add_argument("--toroidal-b", type=float, default=0.0, help="Toroidal background field strength B_phi from central axial current")
    p.add_argument("--toroidal-r0", type=float, default=1.0, help="Core radius used in the toroidal B_phi profile")
    p.add_argument("--dipole-scale-end", type=float, default=None, help="Ramp the dipole scale linearly from --dipole-scale to this value (needs --self-consistent)")
    p.add_argument("--uniform-bz-end", type=float, default=None, help="Ramp the uniform Bz linearly from --uniform-bz to this value (needs --self-consistent)")
    p.add_argument("--ramp-start-step", type=int, default=0, help="Step of this run at which the ramp starts")
    p.add_argument("--ramp-steps", type=int, default=0, help="Ramp duration in steps (0 = ramp over the rest of the run)")
    p.add_argument("--n", type=int, default=96, help="Number of particles")
    p.add_argument("--ring-radius", type=float, default=5.0)
    p.add_argument("--radial-spread", type=float, default=0.3)
    p.add_argument("--z-spread", type=float, default=0.3)
    p.add_argument("--phi-center-deg", type=float, default=None, help="Center azimuth for localized injection")
    p.add_argument("--phi-spread-deg", type=float, default=None, help="Std-dev azimuth spread in degrees")
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--vz-spread", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-mode-m", type=int, default=0, help="Seed an azimuthal density perturbation 1 + A cos(m phi) in the initial particles (0 = off)")
    p.add_argument("--seed-mode-amp", type=float, default=0.0, help="Relative amplitude A of the seeded azimuthal perturbation")
    p.add_argument("--self-consistent", action="store_true", help="Enable self-consistent hybrid update (axisymmetric if --sc-nphi 1)")
    p.add_argument("--sc-nr", type=int, default=96)
    p.add_argument("--sc-nz", type=int, default=128)
    p.add_argument("--sc-nphi", type=int, default=1, help="Azimuthal grid planes (1 = axisymmetric)")
    p.add_argument("--sc-r-min", type=float, default=0.0, help="Inner radius of the grid (use >0 for an annulus that avoids the axis)")
    p.add_argument("--particle-weight", type=float, default=1.0, help="Weight per macro-particle (density = weight*count/volume); use 10000/N to keep the density of the 10k-particle runs")
    p.add_argument("--snapshot3d-stride", type=int, default=0, help="Save full (nr,nphi,nz) float32 snapshots every N steps when --sc-nphi > 1 (0 = off)")
    p.add_argument("--sc-r-max", type=float, default=10.0)
    p.add_argument("--sc-z-max", type=float, default=6.0)
    p.add_argument("--electron-temp", type=float, default=0.05)
    p.add_argument("--resistivity", type=float, default=0.0)
    p.add_argument("--density-floor", type=float, default=1e-4)
    p.add_argument("--with-hall", action="store_true", help="Enable Hall term in Ohm law")
    p.add_argument("--hall-density-floor", type=float, default=0.0, help="Minimum density used in the Hall-term denominator (0 = use --density-floor)")
    p.add_argument("--hall-wdt-max", type=float, default=0.0, help="Cap on whistler omega_max*dt per cell via a |B|-dependent Hall density (0 = off)")
    p.add_argument("--evolve-magnetic", action="store_true", help="Evolve perturbation magnetic field with Faraday")
    p.add_argument("--e-clip", type=float, default=10.0, help="Clip each electric-field component on grid")
    p.add_argument("--j-clip", type=float, default=10.0, help="Clip each current component on grid")
    p.add_argument("--snapshot-stride", type=int, default=20, help="Store self-consistent grid snapshots every N steps")
    p.add_argument("--heartbeat-every", type=int, default=0, help="Print progress every N steps in self-consistent mode")
    p.add_argument("--history-stride", type=int, default=1, help="Store particle histories (x, v, K, mu) only every N steps; the last step is always stored (self-consistent mode)")
    p.add_argument("--restart-input", type=str, default=None, help="Existing NPZ to continue from final state")
    p.add_argument("--output", type=str, default="hybrid/output/ensemble.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ramp_requested = args.dipole_scale_end is not None or args.uniform_bz_end is not None
    if ramp_requested and not args.self_consistent:
        raise SystemExit("--dipole-scale-end / --uniform-bz-end require --self-consistent")
    ramp_steps = args.ramp_steps if args.ramp_steps > 0 else max(args.steps - args.ramp_start_step, 0)
    run_cfg = DipoleRunConfig(
        steps=args.steps,
        dt=args.dt,
        q_over_m=args.q_over_m,
        dipole_moment=(0.0, 0.0, args.dipole_scale),
        with_corotation=args.with_corotation,
        omega_vec=(0.0, 0.0, args.omega),
        uniform_bz=args.uniform_bz,
        outer_bz=args.outer_bz,
        outer_bz_r0=args.outer_bz_r0,
        outer_bz_width=args.outer_bz_width,
        toroidal_b=args.toroidal_b,
        toroidal_r0=args.toroidal_r0,
        dipole_moment_end=(0.0, 0.0, args.dipole_scale_end) if args.dipole_scale_end is not None else None,
        uniform_bz_end=args.uniform_bz_end,
        ramp_start_step=args.ramp_start_step,
        ramp_steps=ramp_steps if ramp_requested else 0,
    )
    ens_cfg = EnsembleConfig(
        n_particles=args.n,
        ring_radius=args.ring_radius,
        radial_spread=args.radial_spread,
        z_spread=args.z_spread,
        phi_center_deg=args.phi_center_deg,
        phi_spread_deg=args.phi_spread_deg,
        speed=args.speed,
        vz_spread=args.vz_spread,
        seed=args.seed,
        seed_mode_m=args.seed_mode_m,
        seed_mode_amp=args.seed_mode_amp,
    )
    sc_cfg = SelfConsistentConfig(
        enabled=args.self_consistent,
        nr=args.sc_nr,
        nz=args.sc_nz,
        r_min=args.sc_r_min,
        r_max=args.sc_r_max,
        z_min=-args.sc_z_max,
        z_max=args.sc_z_max,
        electron_temp=args.electron_temp,
        resistivity=args.resistivity,
        density_floor=args.density_floor,
        include_hall=args.with_hall,
        hall_density_floor=args.hall_density_floor,
        hall_wdt_max=args.hall_wdt_max,
        evolve_magnetic=args.evolve_magnetic,
        e_clip=args.e_clip,
        j_clip=args.j_clip,
        snapshot_stride=args.snapshot_stride,
        save_snapshots=True,
        heartbeat_every=args.heartbeat_every,
        history_stride=args.history_stride,
        nphi=args.sc_nphi,
        particle_weight=args.particle_weight,
        snapshot3d_stride=args.snapshot3d_stride,
    )

    restart_state = None
    if args.restart_input is not None:
        restart = np.load(args.restart_input, allow_pickle=False)
        x_prev = np.asarray(restart["x"], dtype=float)
        v_prev = np.asarray(restart["v"], dtype=float)
        if x_prev.ndim != 3 or v_prev.ndim != 3:
            raise ValueError("restart-input must contain particle histories with shape (n, t, 3)")

        x_last = x_prev[:, -1, :]
        v_last = v_prev[:, -1, :]
        time0 = float(np.asarray(restart["t"], dtype=float)[-1]) if "t" in restart.files else 0.0

        dbr = dbphi = dbz = None
        # Prefer the full (nr, nphi, nz) final fields of a 3D run; otherwise use the last
        # (phi-averaged / axisymmetric) snapshot, which the solver extrudes over phi.
        for name in ("dbr", "dbphi", "dbz"):
            val = None
            if f"last_{name}" in restart.files and restart[f"last_{name}"].size > 0:
                val = np.asarray(restart[f"last_{name}"], dtype=float)
            elif f"snap_{name}" in restart.files and restart[f"snap_{name}"].size > 0:
                val = np.asarray(restart[f"snap_{name}"], dtype=float)[-1]
            if name == "dbr":
                dbr = val
            elif name == "dbphi":
                dbphi = val
            else:
                dbz = val

        restart_state = SelfConsistentRestartState(
            x=x_last,
            v=v_last,
            dbr=dbr,
            dbphi=dbphi,
            dbz=dbz,
            time0=time0,
        )

    result = run_particle_ensemble(run_cfg, ens_cfg, sc_cfg, restart_state=restart_state)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Extra 3D / mode-spectrum outputs (only present for self-consistent runs).
    extra = {
        key: val
        for key, val in result.items()
        if key.startswith(("mode_", "snap3d_", "last_")) or key == "nphi"
    }
    np.savez(
        out_path,
        **extra,
        t=result["t"],
        x=result["x"],
        v=result["v"],
        K=result["K"],
        mu=result["mu"],
        density_mean=result.get("density_mean", np.array([], dtype=float)),
        hist_step=result.get("hist_step", np.array([], dtype=int)),
        r_grid=result.get("r_grid", np.array([], dtype=float)),
        z_grid=result.get("z_grid", np.array([], dtype=float)),
        snap_t_idx=result.get("snap_t_idx", np.array([], dtype=int)),
        snap_n=result.get("snap_n", np.array([], dtype=float)),
        snap_br=result.get("snap_br", np.array([], dtype=float)),
        snap_bz=result.get("snap_bz", np.array([], dtype=float)),
        snap_dbr=result.get("snap_dbr", np.array([], dtype=float)),
        snap_dbz=result.get("snap_dbz", np.array([], dtype=float)),
        snap_dbphi=result.get("snap_dbphi", np.array([], dtype=float)),
        dipole_mz_t=result.get("dipole_mz_t", np.array([], dtype=float)),
        uniform_bz_t=result.get("uniform_bz_t", np.array([], dtype=float)),
        with_corotation=args.with_corotation,
        omega=args.omega,
        self_consistent=args.self_consistent,
    )

    k = result["K"]
    mu = result["mu"]
    k_drift = (k[:, -1] - k[:, 0]) / np.maximum(np.abs(k[:, 0]), 1e-12)
    mu_drift = (mu[:, -1] - mu[:, 0]) / np.maximum(np.abs(mu[:, 0]), 1e-12)
    print(f"saved: {out_path}")
    effective_n = int(result["x"].shape[0])
    print(
        "particles="
        f"{effective_n} steps={args.steps} corotation={args.with_corotation} "
        f"dipole_scale={args.dipole_scale} self_consistent={args.self_consistent}"
    )
    if args.restart_input is not None:
        print(f"restart_input={args.restart_input}")
    print(f"mean |K drift| = {np.mean(np.abs(k_drift)):.3e}")
    print(f"mean |mu drift| = {np.mean(np.abs(mu_drift)):.3e}")


if __name__ == "__main__":
    main()
