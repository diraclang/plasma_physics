from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fields import (
    corotation_e_field,
    dipole_b_field,
    northward_confining_b_field,
    uniform_northward_b_field,
)
from .pusher import boris_step


@dataclass
class DipoleRunConfig:
    steps: int = 20_000
    dt: float = 1e-3
    q_over_m: float = 1.0
    dipole_moment: tuple[float, float, float] = (0.0, 0.0, 1.0)
    omega_vec: tuple[float, float, float] = (0.0, 0.0, 0.05)
    with_corotation: bool = False
    uniform_bz: float = 0.0
    outer_bz: float = 0.0
    outer_bz_r0: float = 8.0
    outer_bz_width: float = 0.8
    x0: tuple[float, float, float] = (5.0, 0.0, 0.2)
    v0: tuple[float, float, float] = (0.0, 0.35, 0.2)


@dataclass
class EnsembleConfig:
    n_particles: int = 64
    ring_radius: float = 5.0
    radial_spread: float = 0.3
    z_spread: float = 0.3
    phi_center_deg: float | None = None
    phi_spread_deg: float | None = None
    speed: float = 0.4
    vz_spread: float = 0.1
    seed: int = 42


@dataclass
class SelfConsistentConfig:
    enabled: bool = False
    nr: int = 96
    nz: int = 128
    r_min: float = 0.0
    r_max: float = 10.0
    z_min: float = -6.0
    z_max: float = 6.0
    electron_temp: float = 0.05
    resistivity: float = 0.0
    density_floor: float = 1e-4
    elementary_charge: float = 1.0
    mu0: float = 1.0
    include_hall: bool = False
    evolve_magnetic: bool = False
    e_clip: float = 10.0
    j_clip: float = 10.0
    snapshot_stride: int = 20
    save_snapshots: bool = True
    heartbeat_every: int = 0


@dataclass
class SelfConsistentRestartState:
    x: np.ndarray
    v: np.ndarray
    dbr: np.ndarray | None = None
    dbphi: np.ndarray | None = None
    dbz: np.ndarray | None = None
    time0: float = 0.0


def run_single_particle(cfg: DipoleRunConfig) -> dict[str, np.ndarray]:
    x = np.array(cfg.x0, dtype=float)
    v = np.array(cfg.v0, dtype=float)
    m = np.array(cfg.dipole_moment, dtype=float)
    omega = np.array(cfg.omega_vec, dtype=float)

    xs = np.zeros((cfg.steps + 1, 3), dtype=float)
    vs = np.zeros((cfg.steps + 1, 3), dtype=float)
    bs = np.zeros((cfg.steps + 1, 3), dtype=float)
    es = np.zeros((cfg.steps + 1, 3), dtype=float)
    kinetic = np.zeros(cfg.steps + 1, dtype=float)
    mu = np.zeros(cfg.steps + 1, dtype=float)

    def record(i: int, x_now: np.ndarray, v_now: np.ndarray) -> None:
        b_now = dipole_b_field(x_now, m) + uniform_northward_b_field(cfg.uniform_bz) + northward_confining_b_field(
            x_now,
            cfg.outer_bz,
            cfg.outer_bz_r0,
            cfg.outer_bz_width,
        )
        if cfg.with_corotation:
            e_now = corotation_e_field(x_now, b_now, omega)
        else:
            e_now = np.zeros(3, dtype=float)

        bmag = np.linalg.norm(b_now)
        bhat = b_now / bmag if bmag > 0 else np.array([0.0, 0.0, 1.0])
        vpar = np.dot(v_now, bhat) * bhat
        vperp = v_now - vpar

        xs[i] = x_now
        vs[i] = v_now
        bs[i] = b_now
        es[i] = e_now
        kinetic[i] = 0.5 * np.dot(v_now, v_now)
        mu[i] = 0.5 * np.dot(vperp, vperp) / max(bmag, 1e-12)

    record(0, x, v)

    for i in range(1, cfg.steps + 1):
        b = dipole_b_field(x, m) + uniform_northward_b_field(cfg.uniform_bz) + northward_confining_b_field(
            x,
            cfg.outer_bz,
            cfg.outer_bz_r0,
            cfg.outer_bz_width,
        )
        if cfg.with_corotation:
            e = corotation_e_field(x, b, omega)
        else:
            e = np.zeros(3, dtype=float)
        x, v = boris_step(x, v, cfg.q_over_m, cfg.dt, e, b)
        record(i, x, v)

    t = np.arange(cfg.steps + 1, dtype=float) * cfg.dt
    return {
        "t": t,
        "x": xs,
        "v": vs,
        "B": bs,
        "E": es,
        "K": kinetic,
        "mu": mu,
    }


def make_ensemble_initial_conditions(
    cfg: EnsembleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create ring-like initial conditions for quick transport studies."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.phi_spread_deg is None or cfg.phi_center_deg is None or cfg.phi_spread_deg >= 360.0:
        phi = rng.uniform(0.0, 2.0 * np.pi, cfg.n_particles)
    else:
        phi0 = np.deg2rad(cfg.phi_center_deg)
        sigma = np.deg2rad(max(cfg.phi_spread_deg, 1e-6))
        phi = phi0 + rng.normal(0.0, sigma, cfg.n_particles)
        phi = np.mod(phi, 2.0 * np.pi)
    r = cfg.ring_radius + rng.normal(0.0, cfg.radial_spread, cfg.n_particles)
    z = rng.normal(0.0, cfg.z_spread, cfg.n_particles)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pos = np.column_stack([x, y, z])

    # Start mostly azimuthal with a small vertical spread.
    ephi = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    vz = rng.normal(0.0, cfg.vz_spread, cfg.n_particles)
    vel = cfg.speed * ephi
    vel[:, 2] += vz
    return pos, vel


def run_particle_ensemble(
    run_cfg: DipoleRunConfig,
    ens_cfg: EnsembleConfig,
    self_cfg: SelfConsistentConfig | None = None,
    restart_state: SelfConsistentRestartState | None = None,
) -> dict[str, np.ndarray]:
    """Run independent test particles and return stacked diagnostics."""
    if self_cfg is not None and self_cfg.enabled:
        return run_particle_ensemble_self_consistent(run_cfg, ens_cfg, self_cfg, restart_state)

    x0s, v0s = make_ensemble_initial_conditions(ens_cfg)
    n = ens_cfg.n_particles
    ntime = run_cfg.steps + 1

    xs = np.zeros((n, ntime, 3), dtype=float)
    vs = np.zeros((n, ntime, 3), dtype=float)
    ks = np.zeros((n, ntime), dtype=float)
    mus = np.zeros((n, ntime), dtype=float)

    for i in range(n):
        cfg_i = DipoleRunConfig(
            steps=run_cfg.steps,
            dt=run_cfg.dt,
            q_over_m=run_cfg.q_over_m,
            dipole_moment=run_cfg.dipole_moment,
            omega_vec=run_cfg.omega_vec,
            with_corotation=run_cfg.with_corotation,
            uniform_bz=run_cfg.uniform_bz,
            outer_bz=run_cfg.outer_bz,
            outer_bz_r0=run_cfg.outer_bz_r0,
            outer_bz_width=run_cfg.outer_bz_width,
            x0=tuple(x0s[i]),
            v0=tuple(v0s[i]),
        )
        out = run_single_particle(cfg_i)
        xs[i] = out["x"]
        vs[i] = out["v"]
        ks[i] = out["K"]
        mus[i] = out["mu"]

    return {
        "t": np.arange(ntime, dtype=float) * run_cfg.dt,
        "x": xs,
        "v": vs,
        "K": ks,
        "mu": mus,
    }


def _grad_axisymmetric(f: np.ndarray, dr: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    ddr = np.empty_like(f)
    ddz = np.empty_like(f)

    ddr[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dr)
    ddr[0, :] = (f[1, :] - f[0, :]) / dr
    ddr[-1, :] = (f[-1, :] - f[-2, :]) / dr

    ddz[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dz)
    ddz[:, 0] = (f[:, 1] - f[:, 0]) / dz
    ddz[:, -1] = (f[:, -1] - f[:, -2]) / dz
    return ddr, ddz


def _curl_axisymmetric(
    ar: np.ndarray,
    aphi: np.ndarray,
    az: np.ndarray,
    r_grid: np.ndarray,
    dr: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dar_dr, dar_dz = _grad_axisymmetric(ar, dr, dz)
    daphi_dr, daphi_dz = _grad_axisymmetric(aphi, dr, dz)
    daz_dr, _daz_dz = _grad_axisymmetric(az, dr, dz)

    r_safe = np.maximum(r_grid[:, None], 0.5 * dr)
    curl_r = -daphi_dz
    curl_phi = dar_dz - daz_dr
    curl_z = (1.0 / r_safe) * (daphi_dr * r_safe + aphi)
    return curl_r, curl_phi, curl_z


def _deposit_axisymmetric_moments(
    pos: np.ndarray,
    vel: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    dr: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr = r_grid.size
    nz = z_grid.size
    wsum = np.zeros((nr, nz), dtype=float)
    ur_sum = np.zeros((nr, nz), dtype=float)
    uphi_sum = np.zeros((nr, nz), dtype=float)
    uz_sum = np.zeros((nr, nz), dtype=float)

    r = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])
    z = pos[:, 2]
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    ur = vel[:, 0] * cosphi + vel[:, 1] * sinphi
    uphi = -vel[:, 0] * sinphi + vel[:, 1] * cosphi
    uz = vel[:, 2]

    fr = (r - r_grid[0]) / dr
    fz = (z - z_grid[0]) / dz
    ir0 = np.floor(fr).astype(int)
    iz0 = np.floor(fz).astype(int)

    valid = (ir0 >= 0) & (ir0 < nr - 1) & (iz0 >= 0) & (iz0 < nz - 1)
    valid_idx = np.nonzero(valid)[0]

    for p in valid_idx:
        i = ir0[p]
        k = iz0[p]
        wr1 = fr[p] - i
        wz1 = fz[p] - k
        wr0 = 1.0 - wr1
        wz0 = 1.0 - wz1

        w00 = wr0 * wz0
        w10 = wr1 * wz0
        w01 = wr0 * wz1
        w11 = wr1 * wz1

        wsum[i, k] += w00
        wsum[i + 1, k] += w10
        wsum[i, k + 1] += w01
        wsum[i + 1, k + 1] += w11

        ur_sum[i, k] += w00 * ur[p]
        ur_sum[i + 1, k] += w10 * ur[p]
        ur_sum[i, k + 1] += w01 * ur[p]
        ur_sum[i + 1, k + 1] += w11 * ur[p]

        uphi_sum[i, k] += w00 * uphi[p]
        uphi_sum[i + 1, k] += w10 * uphi[p]
        uphi_sum[i, k + 1] += w01 * uphi[p]
        uphi_sum[i + 1, k + 1] += w11 * uphi[p]

        uz_sum[i, k] += w00 * uz[p]
        uz_sum[i + 1, k] += w10 * uz[p]
        uz_sum[i, k + 1] += w01 * uz[p]
        uz_sum[i + 1, k + 1] += w11 * uz[p]

    w_safe = np.maximum(wsum, 1e-12)
    uir = ur_sum / w_safe
    uiphi = uphi_sum / w_safe
    uiz = uz_sum / w_safe

    # Axisymmetric ring-cell volume in normalized units.
    volume = 2.0 * np.pi * np.maximum(r_grid[:, None], 0.5 * dr) * dr * dz
    n = wsum / np.maximum(volume, 1e-12)
    return n, uir, uiphi, uiz


def _interp_axisymmetric(
    fr: np.ndarray,
    fphi: np.ndarray,
    fz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    pos: np.ndarray,
    dr: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nr = r_grid.size
    nz = z_grid.size
    out_r = np.zeros(pos.shape[0], dtype=float)
    out_phi = np.zeros(pos.shape[0], dtype=float)
    out_z = np.zeros(pos.shape[0], dtype=float)

    r = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])
    z = pos[:, 2]
    fr_idx = (r - r_grid[0]) / dr
    fz_idx = (z - z_grid[0]) / dz
    ir0 = np.floor(fr_idx).astype(int)
    iz0 = np.floor(fz_idx).astype(int)

    valid = (ir0 >= 0) & (ir0 < nr - 1) & (iz0 >= 0) & (iz0 < nz - 1)
    valid_idx = np.nonzero(valid)[0]

    for p in valid_idx:
        i = ir0[p]
        k = iz0[p]
        wr1 = fr_idx[p] - i
        wz1 = fz_idx[p] - k
        wr0 = 1.0 - wr1
        wz0 = 1.0 - wz1

        w00 = wr0 * wz0
        w10 = wr1 * wz0
        w01 = wr0 * wz1
        w11 = wr1 * wz1

        out_r[p] = w00 * fr[i, k] + w10 * fr[i + 1, k] + w01 * fr[i, k + 1] + w11 * fr[i + 1, k + 1]
        out_phi[p] = (
            w00 * fphi[i, k] + w10 * fphi[i + 1, k] + w01 * fphi[i, k + 1] + w11 * fphi[i + 1, k + 1]
        )
        out_z[p] = w00 * fz[i, k] + w10 * fz[i + 1, k] + w01 * fz[i, k + 1] + w11 * fz[i + 1, k + 1]

    return out_r, out_phi, out_z


def _cylindrical_to_cartesian_vectors(
    vr: np.ndarray,
    vphi: np.ndarray,
    vz: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray:
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    vx = vr * cosphi - vphi * sinphi
    vy = vr * sinphi + vphi * cosphi
    return np.column_stack([vx, vy, vz])


def run_particle_ensemble_self_consistent(
    run_cfg: DipoleRunConfig,
    ens_cfg: EnsembleConfig,
    self_cfg: SelfConsistentConfig,
    restart_state: SelfConsistentRestartState | None = None,
) -> dict[str, np.ndarray]:
    """Run axisymmetric hybrid update: ion particles + electron-fluid Ohm law."""
    if restart_state is None:
        x, v = make_ensemble_initial_conditions(ens_cfg)
        time0 = 0.0
    else:
        x = np.asarray(restart_state.x, dtype=float).copy()
        v = np.asarray(restart_state.v, dtype=float).copy()
        time0 = float(restart_state.time0)
    n_particles = x.shape[0]
    ntime = run_cfg.steps + 1

    r_grid = np.linspace(self_cfg.r_min, self_cfg.r_max, self_cfg.nr)
    z_grid = np.linspace(self_cfg.z_min, self_cfg.z_max, self_cfg.nz)
    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])

    dipole_moment = np.asarray(run_cfg.dipole_moment, dtype=float)
    omega = np.asarray(run_cfg.omega_vec, dtype=float)

    br0 = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
    bphi0 = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
    bz0 = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
    for i, r in enumerate(r_grid):
        for k, z in enumerate(z_grid):
            p = np.array([r, 0.0, z], dtype=float)
            b_cart = (
                dipole_b_field(p, dipole_moment)
                + uniform_northward_b_field(run_cfg.uniform_bz)
                + northward_confining_b_field(
                p,
                run_cfg.outer_bz,
                run_cfg.outer_bz_r0,
                run_cfg.outer_bz_width,
                )
            )
            br0[i, k] = b_cart[0]
            bphi0[i, k] = b_cart[1]
            bz0[i, k] = b_cart[2]

    dbr = np.zeros_like(br0)
    dbphi = np.zeros_like(bphi0)
    dbz = np.zeros_like(bz0)
    if restart_state is not None:
        if restart_state.dbr is not None:
            dbr = np.asarray(restart_state.dbr, dtype=float).copy()
        if restart_state.dbphi is not None:
            dbphi = np.asarray(restart_state.dbphi, dtype=float).copy()
        if restart_state.dbz is not None:
            dbz = np.asarray(restart_state.dbz, dtype=float).copy()

    xs = np.zeros((n_particles, ntime, 3), dtype=float)
    vs = np.zeros((n_particles, ntime, 3), dtype=float)
    ks = np.zeros((n_particles, ntime), dtype=float)
    mus = np.zeros((n_particles, ntime), dtype=float)
    density_mean = np.zeros(ntime, dtype=float)
    wall_t0 = time.perf_counter()

    snap_t_idx: list[int] = []
    snap_n: list[np.ndarray] = []
    snap_br: list[np.ndarray] = []
    snap_bz: list[np.ndarray] = []
    snap_dbr: list[np.ndarray] = []
    snap_dbz: list[np.ndarray] = []
    snap_dbphi: list[np.ndarray] = []

    def record(step: int, pos_now: np.ndarray, vel_now: np.ndarray, e_cart: np.ndarray, b_cart: np.ndarray) -> None:
        xs[:, step, :] = pos_now
        vs[:, step, :] = vel_now
        ks[:, step] = 0.5 * np.sum(vel_now * vel_now, axis=1)

        bmag = np.linalg.norm(b_cart, axis=1)
        bhat = np.divide(b_cart, np.maximum(bmag[:, None], 1e-12))
        vpar = np.sum(vel_now * bhat, axis=1)[:, None] * bhat
        vperp = vel_now - vpar
        mus[:, step] = 0.5 * np.sum(vperp * vperp, axis=1) / np.maximum(bmag, 1e-12)

    for step in range(ntime):
        n, uir, uiphi, uiz = _deposit_axisymmetric_moments(x, v, r_grid, z_grid, dr, dz)
        n_eff = np.maximum(n, self_cfg.density_floor)
        density_mean[step] = float(np.mean(n_eff))

        br = br0 + dbr
        bphi = bphi0 + dbphi
        bz = bz0 + dbz

        if self_cfg.save_snapshots and (
            step == 0 or step == run_cfg.steps or step % max(self_cfg.snapshot_stride, 1) == 0
        ):
            snap_t_idx.append(step)
            snap_n.append(n_eff.copy())
            snap_br.append(br.copy())
            snap_bz.append(bz.copy())
            snap_dbr.append(dbr.copy())
            snap_dbz.append(dbz.copy())
            snap_dbphi.append(dbphi.copy())

        j_r, j_phi, j_z = _curl_axisymmetric(dbr, dbphi, dbz, r_grid, dr, dz)
        j_r /= self_cfg.mu0
        j_phi /= self_cfg.mu0
        j_z /= self_cfg.mu0
        if self_cfg.j_clip > 0.0:
            j_r = np.clip(j_r, -self_cfg.j_clip, self_cfg.j_clip)
            j_phi = np.clip(j_phi, -self_cfg.j_clip, self_cfg.j_clip)
            j_z = np.clip(j_z, -self_cfg.j_clip, self_cfg.j_clip)

        pe = self_cfg.electron_temp * n_eff
        dpe_dr, dpe_dz = _grad_axisymmetric(pe, dr, dz)

        # -u_i x B
        e_r = -(uiphi * bz - uiz * bphi)
        e_phi = -(uiz * br - uir * bz)
        e_z = -(uir * bphi - uiphi * br)

        # Hall term: (J x B) / (e n)
        if self_cfg.include_hall:
            denom = self_cfg.elementary_charge * n_eff
            e_r += (j_phi * bz - j_z * bphi) / denom
            e_phi += (j_z * br - j_r * bz) / denom
            e_z += (j_r * bphi - j_phi * br) / denom

        # Electron pressure term: -grad(p_e)/(e n)
        denom = self_cfg.elementary_charge * n_eff
        e_r += -dpe_dr / denom
        e_z += -dpe_dz / denom

        # Resistive term: eta J
        e_r += self_cfg.resistivity * j_r
        e_phi += self_cfg.resistivity * j_phi
        e_z += self_cfg.resistivity * j_z

        if self_cfg.e_clip > 0.0:
            e_r = np.clip(e_r, -self_cfg.e_clip, self_cfg.e_clip)
            e_phi = np.clip(e_phi, -self_cfg.e_clip, self_cfg.e_clip)
            e_z = np.clip(e_z, -self_cfg.e_clip, self_cfg.e_clip)

        if self_cfg.evolve_magnetic and step < run_cfg.steps:
            curl_er, curl_ephi, curl_ez = _curl_axisymmetric(e_r, e_phi, e_z, r_grid, dr, dz)
            dbr += -run_cfg.dt * curl_er
            dbphi += -run_cfg.dt * curl_ephi
            dbz += -run_cfg.dt * curl_ez

        epr, epphi, epz = _interp_axisymmetric(e_r, e_phi, e_z, r_grid, z_grid, x, dr, dz)
        dbpr, dbpphi, dbpz = _interp_axisymmetric(dbr, dbphi, dbz, r_grid, z_grid, x, dr, dz)

        e_cart_sc = _cylindrical_to_cartesian_vectors(epr, epphi, epz, x)
        db_cart = _cylindrical_to_cartesian_vectors(dbpr, dbpphi, dbpz, x)
        b0_cart = np.array(
            [
                dipole_b_field(p, dipole_moment)
                + uniform_northward_b_field(run_cfg.uniform_bz)
                + northward_confining_b_field(
                    p,
                    run_cfg.outer_bz,
                    run_cfg.outer_bz_r0,
                    run_cfg.outer_bz_width,
                )
                for p in x
            ]
        )
        b_cart = b0_cart + db_cart

        if run_cfg.with_corotation:
            e_coro = np.array([corotation_e_field(p, b, omega) for p, b in zip(x, b_cart, strict=False)])
            e_cart = e_cart_sc + e_coro
        else:
            e_cart = e_cart_sc

        record(step, x, v, e_cart, b_cart)

        if step < run_cfg.steps:
            x_next = np.zeros_like(x)
            v_next = np.zeros_like(v)
            for p in range(n_particles):
                x_next[p], v_next[p] = boris_step(
                    x[p],
                    v[p],
                    run_cfg.q_over_m,
                    run_cfg.dt,
                    e_cart[p],
                    b_cart[p],
                )
            x = x_next
            v = v_next

        if self_cfg.heartbeat_every > 0 and (
            step == 0 or step == run_cfg.steps or step % self_cfg.heartbeat_every == 0
        ):
            elapsed = time.perf_counter() - wall_t0
            frac = step / max(run_cfg.steps, 1)
            k_mean = float(np.mean(ks[:, step]))
            print(
                "[heartbeat] "
                f"step={step}/{run_cfg.steps} "
                f"sim_t={time0 + step * run_cfg.dt:.4f} "
                f"progress={100.0 * frac:.1f}% "
                f"density_mean={density_mean[step]:.4e} "
                f"k_mean={k_mean:.4e} "
                f"wall_s={elapsed:.1f}",
                flush=True,
            )

    return {
        "t": time0 + np.arange(ntime, dtype=float) * run_cfg.dt,
        "x": xs,
        "v": vs,
        "K": ks,
        "mu": mus,
        "density_mean": density_mean,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "snap_t_idx": np.asarray(snap_t_idx, dtype=int),
        "snap_n": np.asarray(snap_n, dtype=float),
        "snap_br": np.asarray(snap_br, dtype=float),
        "snap_bz": np.asarray(snap_bz, dtype=float),
        "snap_dbr": np.asarray(snap_dbr, dtype=float),
        "snap_dbz": np.asarray(snap_dbz, dtype=float),
        "snap_dbphi": np.asarray(snap_dbphi, dtype=float),
    }
