from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fields import (
    corotation_e_field,
    corotation_e_field_many,
    dipole_b_field,
    external_b_field_many,
    northward_confining_b_field,
    toroidal_b_field,
    uniform_northward_b_field,
)
from .pusher import boris_step, boris_step_many
from . import kernels


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
    toroidal_b: float = 0.0
    toroidal_r0: float = 1.0
    # Optional linear ramp of the external field inside a self-consistent run.
    # The ramp runs from the values above to the *_end values over `ramp_steps`
    # steps starting at step `ramp_start_step` of this run (0 ramp_steps = no ramp).
    dipole_moment_end: tuple[float, float, float] | None = None
    uniform_bz_end: float | None = None
    ramp_start_step: int = 0
    ramp_steps: int = 0
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
    # Seed an azimuthal density perturbation n ~ 1 + amp*cos(m*phi) by remapping the
    # particle azimuths (no extra random draws).  0 = off.
    seed_mode_m: int = 0
    seed_mode_amp: float = 0.0


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
    hall_density_floor: float = 0.0
    hall_wdt_max: float = 0.0
    evolve_magnetic: bool = False
    e_clip: float = 10.0
    j_clip: float = 10.0
    snapshot_stride: int = 20
    save_snapshots: bool = True
    heartbeat_every: int = 0
    # Store the per-particle histories (x, v, K, mu) only every `history_stride`
    # steps (the last step is always stored).  1 = every step (original behaviour).
    history_stride: int = 1
    # Number of azimuthal grid planes.  1 = axisymmetric (original behaviour); fields
    # live on an (nr, nphi, nz) grid with periodic phi.
    nphi: int = 1
    # Statistical weight of each macro-particle (density = weight * count / volume).
    # Scale as 1/N to keep the physical density fixed when changing particle number.
    particle_weight: float = 1.0
    # Save full (nr, nphi, nz) snapshots (float32) every N steps when nphi > 1 (0 = off).
    snapshot3d_stride: int = 0
    # Use the parallel Numba kernels for deposit/gather/push when Numba is importable
    # (identical arithmetic to the NumPy path; False forces the NumPy path).
    use_numba: bool = True


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
        b_now = (
            dipole_b_field(x_now, m)
            + uniform_northward_b_field(cfg.uniform_bz)
            + northward_confining_b_field(
                x_now,
                cfg.outer_bz,
                cfg.outer_bz_r0,
                cfg.outer_bz_width,
            )
            + toroidal_b_field(x_now, cfg.toroidal_b, cfg.toroidal_r0)
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
        b = (
            dipole_b_field(x, m)
            + uniform_northward_b_field(cfg.uniform_bz)
            + northward_confining_b_field(
                x,
                cfg.outer_bz,
                cfg.outer_bz_r0,
                cfg.outer_bz_width,
            )
            + toroidal_b_field(x, cfg.toroidal_b, cfg.toroidal_r0)
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
    if cfg.seed_mode_m > 0 and cfg.seed_mode_amp != 0.0:
        # phi -> phi - (A/m) sin(m phi) maps a uniform distribution to 1 + A cos(m phi).
        m_seed = float(cfg.seed_mode_m)
        phi = np.mod(phi - (cfg.seed_mode_amp / m_seed) * np.sin(m_seed * phi), 2.0 * np.pi)
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
            toroidal_b=run_cfg.toroidal_b,
            toroidal_r0=run_cfg.toroidal_r0,
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
    """d/dr (axis 0) and d/dz (last axis); works for (nr, nz) and (nr, nphi, nz) arrays."""
    ddr = np.empty_like(f)
    ddz = np.empty_like(f)

    ddr[1:-1, ...] = (f[2:, ...] - f[:-2, ...]) / (2.0 * dr)
    ddr[0, ...] = (f[1, ...] - f[0, ...]) / dr
    ddr[-1, ...] = (f[-1, ...] - f[-2, ...]) / dr

    ddz[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dz)
    ddz[..., 0] = (f[..., 1] - f[..., 0]) / dz
    ddz[..., -1] = (f[..., -1] - f[..., -2]) / dz
    return ddr, ddz


def _grad_phi(f: np.ndarray, dphi: float) -> np.ndarray:
    """Centred periodic d/dphi along axis 1 of an (nr, nphi, nz) array."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dphi)


def _curl_cylindrical(
    ar: np.ndarray,
    aphi: np.ndarray,
    az: np.ndarray,
    r_grid: np.ndarray,
    dr: float,
    dphi: float | None,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curl of a cylindrical vector field on an (nr, nphi, nz) grid.

    `dphi=None` drops the azimuthal derivatives (axisymmetric, nphi == 1).
    """
    dar_dr, dar_dz = _grad_axisymmetric(ar, dr, dz)
    daphi_dr, daphi_dz = _grad_axisymmetric(aphi, dr, dz)
    daz_dr, _daz_dz = _grad_axisymmetric(az, dr, dz)

    r_safe = np.maximum(r_grid, 0.5 * dr)[:, None, None]
    curl_r = -daphi_dz
    curl_phi = dar_dz - daz_dr
    curl_z = (1.0 / r_safe) * (daphi_dr * r_safe + aphi)
    if dphi is not None:
        curl_r = curl_r + _grad_phi(az, dphi) / r_safe
        curl_z = curl_z - _grad_phi(ar, dphi) / r_safe
    return curl_r, curl_phi, curl_z


@dataclass
class _Stencil:
    """Per-step particle <-> grid geometry, computed once and shared by deposit and gather."""

    sel: np.ndarray       # indices of particles inside the grid
    flat: np.ndarray      # (ncorner, nsel) flat indices into an (nr, nphi, nz) array
    w: np.ndarray         # (ncorner, nsel) interpolation weights
    cosphi: np.ndarray    # (N,) cos(phi) of every particle
    sinphi: np.ndarray    # (N,) sin(phi) of every particle
    n_particles: int
    ncell: int


def _build_stencil(
    pos: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    nphi: int,
    dr: float,
    dz: float,
) -> _Stencil:
    """Cloud-in-cell stencil in (r, phi, z) with periodic linear weighting in phi."""
    nr = r_grid.size
    nz = z_grid.size
    px = pos[:, 0]
    py = pos[:, 1]
    r = np.sqrt(px * px + py * py)
    rho = np.maximum(r, 1e-300)
    on_axis = r <= 0.0
    cosphi = np.where(on_axis, 1.0, px / rho)
    sinphi = np.where(on_axis, 0.0, py / rho)

    fr = (r - r_grid[0]) / dr
    fz = (pos[:, 2] - z_grid[0]) / dz
    ir0 = np.floor(fr).astype(np.intp)
    iz0 = np.floor(fz).astype(np.intp)
    valid = (ir0 >= 0) & (ir0 < nr - 1) & (iz0 >= 0) & (iz0 < nz - 1)
    sel = np.nonzero(valid)[0]

    i = ir0[sel]
    k = iz0[sel]
    wr1 = fr[sel] - i
    wz1 = fz[sel] - k
    wr0 = 1.0 - wr1
    wz0 = 1.0 - wz1
    rz_corners = [
        (i, k, wr0 * wz0),
        (i + 1, k, wr1 * wz0),
        (i, k + 1, wr0 * wz1),
        (i + 1, k + 1, wr1 * wz1),
    ]

    if nphi == 1:
        planes: list[tuple[np.ndarray | int, np.ndarray | None]] = [(0, None)]
    else:
        dphi = 2.0 * np.pi / nphi
        phi = np.mod(np.arctan2(py[sel], px[sel]), 2.0 * np.pi)
        fp = phi / dphi
        ip0 = np.floor(fp).astype(np.intp)
        wp1 = fp - ip0
        ip0 = ip0 % nphi
        planes = [(ip0, 1.0 - wp1), ((ip0 + 1) % nphi, wp1)]

    ncorner = 4 * len(planes)
    flat = np.empty((ncorner, sel.size), dtype=np.intp)
    w = np.empty((ncorner, sel.size), dtype=float)
    c = 0
    for ip, wp in planes:
        for ii, kk, wrz in rz_corners:
            flat[c] = (ii * nphi + ip) * nz + kk
            w[c] = wrz if wp is None else wrz * wp
            c += 1
    return _Stencil(sel, flat, w, cosphi, sinphi, pos.shape[0], nr * nphi * nz)


def _deposit_from_stencil(
    st: _Stencil,
    vel: np.ndarray,
    shape: tuple[int, int, int],
    r_grid: np.ndarray,
    dr: float,
    dz: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density and mean ion velocity (ur, uphi, uz) on an (nr, nphi, nz) grid."""
    nr, nphi, nz = shape
    sel = st.sel
    cos_s = st.cosphi[sel]
    sin_s = st.sinphi[sel]
    ur = vel[sel, 0] * cos_s + vel[sel, 1] * sin_s
    uphi = -vel[sel, 0] * sin_s + vel[sel, 1] * cos_s
    uz = vel[sel, 2]

    flat = st.flat.ravel()

    def scatter(weights: np.ndarray) -> np.ndarray:
        return np.bincount(flat, weights=weights.ravel(), minlength=st.ncell).reshape(shape)

    wsum = scatter(st.w)
    ur_sum = scatter(st.w * ur)
    uphi_sum = scatter(st.w * uphi)
    uz_sum = scatter(st.w * uz)

    return _finalize_moments(wsum, ur_sum, uphi_sum, uz_sum, nphi, r_grid, dr, dz, weight)


def _finalize_moments(
    wsum: np.ndarray,
    ur_sum: np.ndarray,
    uphi_sum: np.ndarray,
    uz_sum: np.ndarray,
    nphi: int,
    r_grid: np.ndarray,
    dr: float,
    dz: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn deposited sums into density and mean ion velocity (ur, uphi, uz)."""
    w_safe = np.maximum(wsum, 1e-12)
    uir = ur_sum / w_safe
    uiphi = uphi_sum / w_safe
    uiz = uz_sum / w_safe

    # Ring-sector cell volume (2 pi / nphi of the full ring) in normalized units.
    volume = (2.0 * np.pi / nphi) * np.maximum(r_grid, 0.5 * dr)[:, None, None] * dr * dz
    n = weight * wsum / np.maximum(volume, 1e-12)
    return n, uir, uiphi, uiz


def _deposit_moments(
    pos: np.ndarray,
    vel: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    nphi: int,
    dr: float,
    dz: float,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cloud-in-cell deposit of density and mean ion velocity on an (nr, nphi, nz) grid."""
    st = _build_stencil(pos, r_grid, z_grid, nphi, dr, dz)
    return _deposit_from_stencil(st, vel, (r_grid.size, nphi, z_grid.size), r_grid, dr, dz, weight)


def _gather_fields(st: _Stencil, fields: list[np.ndarray]) -> np.ndarray:
    """Gather several (nr, nphi, nz) grid fields to the particles at once -> (N, nfields).

    Particles outside the grid receive zero.
    """
    stacked = np.stack([f.reshape(-1) for f in fields], axis=1)   # (ncell, nfields)
    out = np.zeros((st.n_particles, len(fields)), dtype=float)
    vals = stacked[st.flat]                                        # (ncorner, nsel, nfields)
    out[st.sel] = np.einsum("cs,csf->sf", st.w, vals)
    return out


def _cyl_to_cart(vr: np.ndarray, vphi: np.ndarray, vz: np.ndarray, st: _Stencil) -> np.ndarray:
    vx = vr * st.cosphi - vphi * st.sinphi
    vy = vr * st.sinphi + vphi * st.cosphi
    return np.column_stack([vx, vy, vz])


def _phi_field_from_restart(a: np.ndarray, nr: int, nphi: int, nz: int, name: str) -> np.ndarray:
    """Return an (nr, nphi, nz) copy; a 2D (nr, nz) restart field is extruded over phi."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (nr, nz):
            raise ValueError(f"restart field {name} has shape {arr.shape}, expected {(nr, nz)}")
        return np.repeat(arr[:, None, :], nphi, axis=1).copy()
    if arr.shape != (nr, nphi, nz):
        raise ValueError(f"restart field {name} has shape {arr.shape}, expected {(nr, nphi, nz)}")
    return arr.copy()


def _mode_amplitudes(fields: list[np.ndarray], r_weight: np.ndarray, nphi: int) -> np.ndarray:
    """RMS azimuthal-mode amplitudes m = 0..nphi//2 of (nr, nphi, nz) fields.

    For m >= 1 the value is the amplitude of the cos/sin component (a field
    f0 * (1 + A cos(m phi)) gives m=0: f0 and m: f0 * A).  The (r, z) average is
    volume (r) weighted and the fields are combined in quadrature.
    """
    power = 0.0
    for f in fields:
        spec = np.fft.rfft(f, axis=1) / nphi
        power = power + np.abs(spec) ** 2
    num = np.tensordot(r_weight, power.sum(axis=2), axes=(0, 0))
    amp = np.sqrt(num / (r_weight.sum() * fields[0].shape[2]))
    amp[1:] *= 2.0
    if nphi % 2 == 0 and nphi > 1:
        amp[-1] *= 0.5
    return amp


def run_particle_ensemble_self_consistent(
    run_cfg: DipoleRunConfig,
    ens_cfg: EnsembleConfig,
    self_cfg: SelfConsistentConfig,
    restart_state: SelfConsistentRestartState | None = None,
) -> dict[str, np.ndarray]:
    """Run the hybrid update: ion particles + electron-fluid Ohm law on an (r, phi, z) grid.

    With `self_cfg.nphi == 1` the fields are axisymmetric; otherwise they are resolved on
    `nphi` periodic azimuthal planes (the background field stays axisymmetric).
    """
    if restart_state is None:
        x, v = make_ensemble_initial_conditions(ens_cfg)
        time0 = 0.0
    else:
        x = np.asarray(restart_state.x, dtype=float).copy()
        v = np.asarray(restart_state.v, dtype=float).copy()
        time0 = float(restart_state.time0)
    n_particles = x.shape[0]
    ntime = run_cfg.steps + 1

    nphi = int(self_cfg.nphi)
    if nphi < 1:
        raise ValueError("nphi must be >= 1")
    dphi = 2.0 * np.pi / nphi
    dphi_op = dphi if nphi > 1 else None
    p_weight = float(self_cfg.particle_weight)
    use_numba = bool(self_cfg.use_numba) and kernels.HAVE_NUMBA
    dep_buf = (
        np.empty((kernels.num_chunks(), 4, self_cfg.nr * nphi * self_cfg.nz), dtype=float)
        if use_numba
        else None
    )

    r_grid = np.linspace(self_cfg.r_min, self_cfg.r_max, self_cfg.nr)
    z_grid = np.linspace(self_cfg.z_min, self_cfg.z_max, self_cfg.nz)
    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])
    r_safe3 = np.maximum(r_grid, 0.5 * dr)[:, None, None]

    dipole_moment = np.asarray(run_cfg.dipole_moment, dtype=float)
    omega = np.asarray(run_cfg.omega_vec, dtype=float)

    def background_grid(m_vec: np.ndarray, ubz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        br_g = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
        bphi_g = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
        bz_g = np.zeros((self_cfg.nr, self_cfg.nz), dtype=float)
        for i, r in enumerate(r_grid):
            for k, z in enumerate(z_grid):
                p = np.array([r, 0.0, z], dtype=float)
                b_cart = (
                    dipole_b_field(p, m_vec)
                    + uniform_northward_b_field(ubz)
                    + northward_confining_b_field(
                        p,
                        run_cfg.outer_bz,
                        run_cfg.outer_bz_r0,
                        run_cfg.outer_bz_width,
                    )
                    + toroidal_b_field(p, run_cfg.toroidal_b, run_cfg.toroidal_r0)
                )
                br_g[i, k] = b_cart[0]
                bphi_g[i, k] = b_cart[1]
                bz_g[i, k] = b_cart[2]
        return br_g, bphi_g, bz_g

    br0, bphi0, bz0 = background_grid(dipole_moment, run_cfg.uniform_bz)

    # Optional external-field ramp (quasi-static: the background field changes in
    # time but no inductive electric field is added).
    ramping = run_cfg.ramp_steps > 0 and (
        run_cfg.dipole_moment_end is not None or run_cfg.uniform_bz_end is not None
    )
    m_start = dipole_moment
    m_end = (
        np.asarray(run_cfg.dipole_moment_end, dtype=float)
        if run_cfg.dipole_moment_end is not None
        else dipole_moment
    )
    ubz_start = run_cfg.uniform_bz
    ubz_end = run_cfg.uniform_bz_end if run_cfg.uniform_bz_end is not None else run_cfg.uniform_bz
    if ramping:
        br0_end, bphi0_end, bz0_end = background_grid(m_end, ubz_end)

    dbr = np.zeros((self_cfg.nr, nphi, self_cfg.nz), dtype=float)
    dbphi = np.zeros_like(dbr)
    dbz = np.zeros_like(dbr)
    if restart_state is not None:
        if restart_state.dbr is not None:
            dbr = _phi_field_from_restart(restart_state.dbr, self_cfg.nr, nphi, self_cfg.nz, "dbr")
        if restart_state.dbphi is not None:
            dbphi = _phi_field_from_restart(restart_state.dbphi, self_cfg.nr, nphi, self_cfg.nz, "dbphi")
        if restart_state.dbz is not None:
            dbz = _phi_field_from_restart(restart_state.dbz, self_cfg.nr, nphi, self_cfg.nz, "dbz")

    stride_h = max(int(self_cfg.history_stride), 1)
    hist_steps = [s for s in range(ntime) if s % stride_h == 0 or s == ntime - 1]
    hist_index = {s: i for i, s in enumerate(hist_steps)}
    n_hist = len(hist_steps)
    xs = np.zeros((n_particles, n_hist, 3), dtype=float)
    vs = np.zeros((n_particles, n_hist, 3), dtype=float)
    ks = np.zeros((n_particles, n_hist), dtype=float)
    mus = np.zeros((n_particles, n_hist), dtype=float)
    k_mean_hist = np.zeros(ntime, dtype=float)
    density_mean = np.zeros(ntime, dtype=float)
    dipole_mz_t = np.zeros(ntime, dtype=float)
    uniform_bz_t = np.zeros(ntime, dtype=float)
    wall_t0 = time.perf_counter()

    snap_t_idx: list[int] = []
    snap_n: list[np.ndarray] = []
    snap_br: list[np.ndarray] = []
    snap_bz: list[np.ndarray] = []
    snap_dbr: list[np.ndarray] = []
    snap_dbz: list[np.ndarray] = []
    snap_dbphi: list[np.ndarray] = []

    def phi_mean(a: np.ndarray) -> np.ndarray:
        return a[:, 0, :].copy() if nphi == 1 else a.mean(axis=1)

    n_modes = nphi // 2 + 1
    mode_n_t = np.zeros((ntime, n_modes), dtype=float)
    mode_db_t = np.zeros((ntime, n_modes), dtype=float)
    stride3d = int(self_cfg.snapshot3d_stride) if nphi > 1 else 0
    snap3d_t_idx: list[int] = []
    snap3d: dict[str, list[np.ndarray]] = {"n": [], "dbr": [], "dbphi": [], "dbz": []}
    def record(step: int, pos_now: np.ndarray, vel_now: np.ndarray, e_cart: np.ndarray, b_cart: np.ndarray) -> None:
        k_now = 0.5 * np.sum(vel_now * vel_now, axis=1)
        k_mean_hist[step] = float(np.mean(k_now))
        h = hist_index.get(step)
        if h is None:
            return
        xs[:, h, :] = pos_now
        vs[:, h, :] = vel_now
        ks[:, h] = k_now

        bmag = np.linalg.norm(b_cart, axis=1)
        bhat = np.divide(b_cart, np.maximum(bmag[:, None], 1e-12))
        vpar = np.sum(vel_now * bhat, axis=1)[:, None] * bhat
        vperp = vel_now - vpar
        mus[:, h] = 0.5 * np.sum(vperp * vperp, axis=1) / np.maximum(bmag, 1e-12)

    for step in range(ntime):
        if use_numba:
            x = np.ascontiguousarray(x, dtype=float)
            v = np.ascontiguousarray(v, dtype=float)
            sums = kernels.deposit(
                x, v, float(r_grid[0]), float(z_grid[0]), dr, dz, self_cfg.nr, nphi, self_cfg.nz, dep_buf
            )
            shape3 = (self_cfg.nr, nphi, self_cfg.nz)
            n, uir, uiphi, uiz = _finalize_moments(
                sums[0].reshape(shape3), sums[1].reshape(shape3), sums[2].reshape(shape3),
                sums[3].reshape(shape3), nphi, r_grid, dr, dz, p_weight,
            )
        else:
            stencil = _build_stencil(x, r_grid, z_grid, nphi, dr, dz)
            n, uir, uiphi, uiz = _deposit_from_stencil(
                stencil, v, (self_cfg.nr, nphi, self_cfg.nz), r_grid, dr, dz, p_weight
            )
        n_eff = np.maximum(n, self_cfg.density_floor)
        density_mean[step] = float(np.mean(n_eff))
        mode_n_t[step] = _mode_amplitudes([n_eff], r_safe3[:, 0, 0], nphi)
        mode_db_t[step] = _mode_amplitudes([dbr, dbphi, dbz], r_safe3[:, 0, 0], nphi)

        if ramping:
            s_ramp = min(max((step - run_cfg.ramp_start_step) / run_cfg.ramp_steps, 0.0), 1.0)
            m_now = m_start + s_ramp * (m_end - m_start)
            ubz_now = ubz_start + s_ramp * (ubz_end - ubz_start)
            br0_t = br0 + s_ramp * (br0_end - br0)
            bphi0_t = bphi0 + s_ramp * (bphi0_end - bphi0)
            bz0_t = bz0 + s_ramp * (bz0_end - bz0)
        else:
            m_now = dipole_moment
            ubz_now = run_cfg.uniform_bz
            br0_t, bphi0_t, bz0_t = br0, bphi0, bz0
        dipole_mz_t[step] = m_now[2]
        uniform_bz_t[step] = ubz_now

        br = br0_t[:, None, :] + dbr
        bphi = bphi0_t[:, None, :] + dbphi
        bz = bz0_t[:, None, :] + dbz

        if self_cfg.save_snapshots and (
            step == 0 or step == run_cfg.steps or step % max(self_cfg.snapshot_stride, 1) == 0
        ):
            snap_t_idx.append(step)
            snap_n.append(phi_mean(n_eff))
            snap_br.append(phi_mean(br))
            snap_bz.append(phi_mean(bz))
            snap_dbr.append(phi_mean(dbr))
            snap_dbz.append(phi_mean(dbz))
            snap_dbphi.append(phi_mean(dbphi))

        if stride3d > 0 and (step == 0 or step == run_cfg.steps or step % stride3d == 0):
            snap3d_t_idx.append(step)
            snap3d["n"].append(n_eff.astype(np.float32))
            snap3d["dbr"].append(dbr.astype(np.float32))
            snap3d["dbphi"].append(dbphi.astype(np.float32))
            snap3d["dbz"].append(dbz.astype(np.float32))

        j_r, j_phi, j_z = _curl_cylindrical(dbr, dbphi, dbz, r_grid, dr, dphi_op, dz)
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
            n_hall = np.maximum(n_eff, self_cfg.hall_density_floor)
            if self_cfg.hall_wdt_max > 0.0:
                # Whistler limiter: raise the density seen by the Hall term until
                # omega_max*dt <= hall_wdt_max, with omega_max = k_max^2 |B| / (mu0 e n)
                # and k_max^2 = pi^2 (1/dr^2 + 1/dz^2 [+ 1/(r dphi)^2 when nphi > 1]).
                bmag = np.sqrt(br * br + bphi * bphi + bz * bz)
                k2_max = np.pi**2 * (1.0 / dr**2 + 1.0 / dz**2)
                if nphi > 1:
                    k2_max = k2_max + np.pi**2 / (r_safe3 * dphi) ** 2
                n_whistler = (
                    k2_max * run_cfg.dt * bmag
                    / (self_cfg.mu0 * self_cfg.elementary_charge * self_cfg.hall_wdt_max)
                )
                n_hall = np.maximum(n_hall, n_whistler)
            denom = self_cfg.elementary_charge * n_hall
            e_r += (j_phi * bz - j_z * bphi) / denom
            e_phi += (j_z * br - j_r * bz) / denom
            e_z += (j_r * bphi - j_phi * br) / denom

        # Electron pressure term: -grad(p_e)/(e n)
        denom = self_cfg.elementary_charge * n_eff
        e_r += -dpe_dr / denom
        e_z += -dpe_dz / denom
        if nphi > 1:
            e_phi += -(_grad_phi(pe, dphi) / r_safe3) / denom

        # Resistive term: eta J
        e_r += self_cfg.resistivity * j_r
        e_phi += self_cfg.resistivity * j_phi
        e_z += self_cfg.resistivity * j_z

        if self_cfg.e_clip > 0.0:
            e_r = np.clip(e_r, -self_cfg.e_clip, self_cfg.e_clip)
            e_phi = np.clip(e_phi, -self_cfg.e_clip, self_cfg.e_clip)
            e_z = np.clip(e_z, -self_cfg.e_clip, self_cfg.e_clip)

        if self_cfg.evolve_magnetic and step < run_cfg.steps:
            curl_er, curl_ephi, curl_ez = _curl_cylindrical(e_r, e_phi, e_z, r_grid, dr, dphi_op, dz)
            dbr += -run_cfg.dt * curl_er
            dbphi += -run_cfg.dt * curl_ephi
            dbz += -run_cfg.dt * curl_ez

        if use_numba:
            fields6 = np.stack([e_r, e_phi, e_z, dbr, dbphi, dbz], axis=-1)
            e_cart = np.empty_like(x)
            b_cart = np.empty_like(x)
            x_next = np.empty_like(x)
            v_next = np.empty_like(v)
            kernels.push(
                x, v, fields6, float(r_grid[0]), float(z_grid[0]), dr, dz,
                self_cfg.nr, nphi, self_cfg.nz,
                np.asarray(m_now, dtype=float), float(ubz_now),
                float(run_cfg.outer_bz), float(run_cfg.outer_bz_r0), float(run_cfg.outer_bz_width),
                float(run_cfg.toroidal_b), float(run_cfg.toroidal_r0),
                bool(run_cfg.with_corotation), omega, float(run_cfg.q_over_m), float(run_cfg.dt),
                e_cart, b_cart, x_next, v_next,
            )
        else:
            gathered = _gather_fields(stencil, [e_r, e_phi, e_z, dbr, dbphi, dbz])

            e_cart_sc = _cyl_to_cart(gathered[:, 0], gathered[:, 1], gathered[:, 2], stencil)
            db_cart = _cyl_to_cart(gathered[:, 3], gathered[:, 4], gathered[:, 5], stencil)
            b0_cart = external_b_field_many(
                x,
                m_now,
                ubz_now,
                run_cfg.outer_bz,
                run_cfg.outer_bz_r0,
                run_cfg.outer_bz_width,
                run_cfg.toroidal_b,
                run_cfg.toroidal_r0,
            )
            b_cart = b0_cart + db_cart

            if run_cfg.with_corotation:
                e_cart = e_cart_sc + corotation_e_field_many(x, b_cart, omega)
            else:
                e_cart = e_cart_sc

        record(step, x, v, e_cart, b_cart)

        if step < run_cfg.steps:
            if use_numba:
                x, v = x_next, v_next
            else:
                x, v = boris_step_many(x, v, run_cfg.q_over_m, run_cfg.dt, e_cart, b_cart)

        if self_cfg.heartbeat_every > 0 and (
            step == 0 or step == run_cfg.steps or step % self_cfg.heartbeat_every == 0
        ):
            elapsed = time.perf_counter() - wall_t0
            frac = step / max(run_cfg.steps, 1)
            k_mean = float(k_mean_hist[step])
            ramp_msg = (
                f"dipole_mz={m_now[2]:.3f} uniform_bz={ubz_now:.4f} " if ramping else ""
            )
            print(
                "[heartbeat] "
                f"step={step}/{run_cfg.steps} "
                f"sim_t={time0 + step * run_cfg.dt:.4f} "
                f"progress={100.0 * frac:.1f}% "
                f"density_mean={density_mean[step]:.4e} "
                f"k_mean={k_mean:.4e} "
                f"{ramp_msg}"
                f"wall_s={elapsed:.1f}",
                flush=True,
            )

    t_all = time0 + np.arange(ntime, dtype=float) * run_cfg.dt
    out = {
        "t": t_all[hist_steps],
        "x": xs,
        "v": vs,
        "K": ks,
        "mu": mus,
        "density_mean": density_mean,
        "hist_step": np.asarray(hist_steps, dtype=int),
        "dipole_mz_t": dipole_mz_t,
        "uniform_bz_t": uniform_bz_t,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "snap_t_idx": np.asarray(snap_t_idx, dtype=int),
        "snap_n": np.asarray(snap_n, dtype=float),
        "snap_br": np.asarray(snap_br, dtype=float),
        "snap_bz": np.asarray(snap_bz, dtype=float),
        "snap_dbr": np.asarray(snap_dbr, dtype=float),
        "snap_dbz": np.asarray(snap_dbz, dtype=float),
        "snap_dbphi": np.asarray(snap_dbphi, dtype=float),
        "nphi": np.asarray(nphi, dtype=int),
        "mode_n_t": mode_n_t,
        "mode_db_t": mode_db_t,
    }
    if nphi > 1:
        # Full final perturbation fields so a later run can restart with the phi structure.
        out["last_dbr"] = dbr
        out["last_dbphi"] = dbphi
        out["last_dbz"] = dbz
    if snap3d_t_idx:
        out["snap3d_t_idx"] = np.asarray(snap3d_t_idx, dtype=int)
        for key, arrs in snap3d.items():
            out[f"snap3d_{key}"] = np.asarray(arrs, dtype=np.float32)
    return out
