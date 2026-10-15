"""Coupled field-particle loop for the mixed x-grid / Fourier-mode PIC model.

Representation
--------------
x is a periodic real grid (collocated nodes x_i = i*dx).  y,z are represented
by a bank of retained Fourier harmonics (k_y,k_z).  Every real field is

    F(x,y,z) = F_eq(x) + 2 Re sum_{k in half space} F_k(x) exp(i(k_y y + k_z z))

Modes (m,n) and (-m,-n) are the same real mode, so the configured list is
folded to a half space.  The mean-square over y,z is

    <F_pert^2>_{y,z} = 2 sum_{k in half space} |F_k|^2.

The equilibrium (m=n=0) magnetic field B_eq(x) = (0, Bg, Bz_eq(x)) is a static
external field for the particles.  It carries no perturbation energy and does
no work, so the conserved quantity is

    W = W_kinetic + W_E(perturbation) + W_B(perturbation).

Equations solved per retained mode (code units, eps_0 mu_0 c^2 = 1, Lorentz
force q(E + v x B), so Omega_ce = e B / m_e with no 1/c)
---------------------------------------------------------------------------
Deposition (CIC in x, exact Fourier phase in y,z; w = macroparticle weight):
    rho_k(x_i) = sum_j q w S(x_i - x_j) exp(-i k.r_j) / (dx Ly Lz)
    J_k(x_i)   = sum_j q w v_j S(x_i - x_j) exp(-i k.r_j) / (dx Ly Lz)
Longitudinal field (energy-conserving form: the force is the exact gradient
of the CIC-interpolated potential):
    (k_perp^2 - Delta_c) phi_k = rho_k / eps_0     (Delta_c: 3-point Laplacian)
    E_Lx = -(phi_{i+1} - phi_i)/dx inside cell i,  E_Ly,z = -i k_{y,z} phi(x)
    U_L  = 1/2 int rho phi  (this is the longitudinal electric energy)
Transverse current (central-difference D, E_T is div-free in the same sense):
    (D^2 - k_perp^2) psi = div J_k,  J_T = J_k - grad psi
Transverse fields (leapfrog):
    dB/dt   = -curl E_T
    dE_T/dt = c^2 curl B - J_T / eps_0
Particles: Boris push in E = E_L + E_T, B = B_eq + B_pert, gathered with the
same CIC weights, summing over all modes at each particle position.

Time levels (leapfrog): x^n, v^{n-1/2}, E_T^n, B^{n-1/2}; J^{n+1/2} is
deposited at the mid-step position with v^{n+1/2}.  Energies at time n use
the time-centred forms  W_K = sum 1/2 m v^{n-1/2}.v^{n+1/2},
W_B = 1/(2 mu_0) B^{n-1/2}.B^{n+1/2},  W_E = U_L + eps_0/2 |E_T^n|^2.
(E_L is curl-free and orthogonal to the div-free E_T, so the two energies add.)

The grid operators are circulant in x, so the Poisson / projection solves are
done exactly in x-Fourier space.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

from config import ModeSpec, SimulationConfig
from equilibrium import Bz_eq, derive_units as derive_cfg_units
from fields_transverse import curl_mode
from particles import ParticleSet, load_species


# ---------------------------------------------------------------------------
# mode bank
# ---------------------------------------------------------------------------
@dataclass
class _Mode:
    m: int
    n: int
    ky: float
    kz: float
    kperp2: float
    s: np.ndarray        # sin(kx dx)/dx            (central-difference symbol)
    denom: np.ndarray    # s^2 + kperp2             (projection operator)
    denomL: np.ndarray   # (2-2cos(kx dx))/dx^2 + kperp2   (Poisson operator)
    ET: np.ndarray       # (3, nx) complex, at time n
    B: np.ndarray        # (3, nx) complex, at time n-1/2
    seed: float
    invL: np.ndarray = None   # 1/denomL (0 where singular)
    inv: np.ndarray = None    # 1/denom  (0 where singular)
    mult: float = 2.0         # 2 for half-space (+-k) modes, 1 for the mean (0,0) mode
    is_mean: bool = False


def canonical_modes(cfg: SimulationConfig):
    """Fold the configured (m,n) list to a half space; origin excluded."""
    kept = {}
    for md in cfg.modes:
        if md.m == 0 and md.n == 0:
            continue
        if (-md.m, -md.n) in kept:
            continue
        kept[(md.m, md.n)] = md
    return list(kept.values())


def _ddx(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * dx)


def _safe_inv(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    nz = a > 1.0e-12
    out[nz] = 1.0 / a[nz]
    return out


def build_modes(cfg: SimulationConfig, include_mean: bool = True):
    """Half-space perturbation modes, plus the dynamic (0,0) mean mode by
    default.  The mean channel represents the full equilibrium B_eq + deviations
    in the x-grid field, and it is the correct default for the double-Harris
    equilibrium used here.

    Passing include_mean=False is retained for compatibility with any truly
    static-background runs, but the default behavior is to evolve the mean mode.
    """
    dx = cfg.Lx / cfg.nx
    x = np.arange(cfg.nx) * dx
    kx = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    # Exact x-Fourier symbol for the periodic direction. The mixed Fourier
    # representation keeps the y,z dependence as retained modes but uses the
    # exact spectral derivative in x, matching the intended k-space Maxwell
    # update and avoiding the finite-difference x-stencil from the older solver.
    kx2 = kx * kx
    modes = []
    for md in canonical_modes(cfg):
        ky = cfg.k_y(md)
        kz = cfg.k_z(md)
        kperp2 = ky * ky + kz * kz
        # seed: B = curl A with A = (0, Ay(x), 0)  =>  div B = 0 exactly
        Ay = (md.seed_amplitude * cfg.B0 * cfg.Lx / (2.0 * np.pi)
              * np.sin(2.0 * np.pi * x / cfg.Lx)).astype(np.complex128)
        B = np.zeros((3, cfg.nx), dtype=np.complex128)
        B[0] = -1j * kz * Ay
        B[2] = _ddx(Ay, dx)
        modes.append(_Mode(m=md.m, n=md.n, ky=ky, kz=kz, kperp2=kperp2, s=1j * kx,
                           denom=kx2 + kperp2, denomL=kx2 + kperp2,
                           ET=np.zeros((3, cfg.nx), dtype=np.complex128),
                           B=B, seed=md.seed_amplitude))
    if include_mean:
        B0 = np.zeros((3, cfg.nx), dtype=np.complex128)
        B0[1] = cfg.Bg
        B0[2] = Bz_eq(x, cfg)
        modes.append(_Mode(m=0, n=0, ky=0.0, kz=0.0, kperp2=0.0, s=1j * kx,
                           denom=kx2, denomL=kx2,
                           ET=np.zeros((3, cfg.nx), dtype=np.complex128),
                           B=B0, seed=0.0, mult=1.0, is_mean=True))
    for md in modes:
        md.invL = _safe_inv(md.denomL)
        md.inv = _safe_inv(md.denom)
    return modes


# ---------------------------------------------------------------------------
# time step
# ---------------------------------------------------------------------------
def stable_dt(cfg: SimulationConfig, units) -> float:
    """Timestep from physical resolution limits (cfg.dt is an upper bound):

        Omega_ce_max * dt <= 0.1     (Boris gyro-phase resolution)
        omega_pe_max * dt <= 0.2     (plasma oscillation)
        c * dt / dx       <= 0.5     (light-wave CFL, central differences)
    """
    dx = cfg.Lx / cfg.nx
    b_max = float(np.hypot(cfg.B0, cfg.Bg))
    om_ce = units.e * b_max / units.m_e
    om_pe = units.omega_pe * np.sqrt(1.0 + cfg.n_bg)
    return float(min(cfg.dt, 0.1 / om_ce, 0.2 / om_pe, 0.5 * dx / cfg.c))


# ---------------------------------------------------------------------------
# CIC deposition / gather, field solves, Boris
# ---------------------------------------------------------------------------
def _cic(x: np.ndarray, dx: float, nx: int):
    s = x / dx
    fl = np.floor(s)
    frac = s - fl
    i0 = fl.astype(np.int64) % nx
    return i0, (i0 + 1) % nx, frac


def _deposit(i0, i1, frac, vals: np.ndarray, nx: int) -> np.ndarray:
    re = (np.bincount(i0, weights=(1.0 - frac) * vals.real, minlength=nx)
          + np.bincount(i1, weights=frac * vals.real, minlength=nx))
    im = (np.bincount(i0, weights=(1.0 - frac) * vals.imag, minlength=nx)
          + np.bincount(i1, weights=frac * vals.imag, minlength=nx))
    return re + 1j * im


def _phase(mode: _Mode, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.exp(1j * (mode.ky * y + mode.kz * z))


def deposit_rho(species, cfg, modes, dx, shapes, phases):
    """rho_k(x) for every mode at the current (time n) positions."""
    rho = [np.zeros(cfg.nx, dtype=np.complex128) for _ in modes]
    for sp, (i0, i1, f), ph in zip(species, shapes, phases):
        coef = sp.q * sp.weight / dx
        for km in range(len(modes)):
            rho[km] += _deposit(i0, i1, f, coef * np.conj(ph[km]), cfg.nx)
    return rho


def solve_potential(rho_k, mode: _Mode, eps_0: float):
    """(k_perp^2 - Delta_c) phi = rho / eps_0 with the 3-point Laplacian."""
    return np.fft.ifft(np.fft.fft(rho_k) * mode.invL / eps_0)


def gather_longitudinal(modes, phis, shape, phases, dx):
    """E_L at the particles as the exact gradient of the CIC-interpolated phi."""
    i0, i1, f = shape
    acc = np.zeros((3, i0.shape[0]), dtype=np.complex128)
    for md, phi, ph in zip(modes, phis, phases):
        phi_p = (1.0 - f) * phi[i0] + f * phi[i1]
        acc[0] += -md.mult * (phi[i1] - phi[i0]) / dx * ph
        acc[1] += -1j * md.mult * md.ky * phi_p * ph
        acc[2] += -1j * md.mult * md.kz * phi_p * ph
    return acc.real


def transverse_current(J, mode: _Mode, dx: float):
    """Remove the longitudinal (grad psi) part of J using the exact spectral
    x-derivative.  This matches the exact Fourier representation used in the
    rest of the mode solve.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(J.shape[1], d=dx)
    dJx = np.fft.ifft(1j * kx * np.fft.fft(J[0]))
    div = dJx + 1j * mode.ky * J[1] + 1j * mode.kz * J[2]
    psi = -np.fft.ifft(np.fft.fft(div) * mode.inv)
    dpsi = np.fft.ifft(1j * kx * np.fft.fft(psi))
    JL = np.array([dpsi, 1j * mode.ky * psi, 1j * mode.kz * psi])
    return J - JL


def gather(fields6, shape, phases, mults):
    """sum_k mult_k Re[F_k(x_i) e^{i k.r_i}] over all retained modes."""
    i0, i1, f = shape
    acc = np.zeros((3, i0.shape[0]), dtype=np.complex128)
    for F, ph, mu in zip(fields6, phases, mults):
        acc += mu * ((1.0 - f) * F[:, i0] + f * F[:, i1]) * ph
    return acc.real


def boris(v, E, B, q_over_m, dt):
    h = 0.5 * q_over_m * dt
    vm = v + h * E
    t = h * B
    s = 2.0 * t / (1.0 + np.sum(t * t, axis=0))
    vp = vm + np.cross(vm, t, axis=0)
    vplus = vm + np.cross(vp, s, axis=0)
    return vplus + h * E


def _mode_energy(a, dx, vperp, coef, b=None, mult=2.0, mean_mode: bool = False) -> float:
    """Mode-space energy contribution for one Fourier harmonic.

    In this formulation we accumulate the energy directly from the retained x-
    profile of each mode and sum over modes. The y,z box area is not applied
    again; it is already implicit in the fact that we keep only the x-dependent
    mode amplitudes themselves.
    """
    if b is None:
        return coef * mult * dx * float(np.sum(np.abs(a) ** 2))
    return coef * mult * dx * float(np.sum((a * np.conj(b)).real))


def _background_magnetic_energy(cfg: SimulationConfig, mu_0: float) -> float:
    """Static equilibrium magnetic energy, B_eq(x) = (0, Bg, Bz_eq(x)).

    This is the x-only mean-field contribution in the retained (0,0) channel:
    the field is a 1D profile along x, so we sum over x and do not multiply by
    the transverse box area again.
    """
    dx = cfg.Lx / cfg.nx
    x = np.arange(cfg.nx) * dx
    B0 = np.zeros((3, cfg.nx), dtype=np.float64)
    B0[1] = cfg.Bg
    B0[2] = Bz_eq(x, cfg)
    return 0.5 / mu_0 * dx * float(np.sum(np.abs(B0) ** 2))


def _temperature(sp, v, cfg, nb=32) -> float:
    """Thermal energy per particle: m/3 * sum_a Var_a, variance taken about the
    local (x-binned) mean flow so the Harris drift is not counted as heat."""
    b = np.minimum((sp.x / cfg.Lx * nb).astype(np.int64), nb - 1)
    n = np.bincount(b, minlength=nb).astype(float)
    ok = n > 0
    tot = 0.0
    for a in range(3):
        s1 = np.bincount(b, weights=v[a], minlength=nb)
        s2 = np.bincount(b, weights=v[a] ** 2, minlength=nb)
        tot += float(np.sum(s2[ok] - s1[ok] ** 2 / n[ok])) / v.shape[1]
    return sp.m * tot / 3.0


# ---------------------------------------------------------------------------
# double-sheet symmetry  Phi: (x,y,z) -> (x + Lx/2, -y, z)
#   polar E : Ey -> -Ey ;  pseudovector B : Bx, Bz -> -B, By -> +By (guide kept)
#   v : vy -> -vy ;  J : Jy -> -Jy.   Bz_eq(x+Lx/2) = -Bz_eq(x), By = Bg.
# Fields obey F_(m,n)(x + Lx/2) = S F_(-m,n)(x)  (S = component signs above).
# ---------------------------------------------------------------------------
_S_E = np.array([1.0, -1.0, 1.0])[:, None]
_S_B = np.array([-1.0, 1.0, -1.0])[:, None]
_VY_FLIP = np.array([1.0, -1.0, 1.0])[:, None]


def _image(sp, cfg: SimulationConfig):
    """Image of a particle set under Phi (used for deposition only)."""
    return SimpleNamespace(x=np.mod(sp.x + 0.5 * cfg.Lx, cfg.Lx),
                           y=np.mod(-sp.y, cfg.Ly), z=sp.z,
                           vx=sp.vx, vy=-sp.vy, vz=sp.vz,
                           q=sp.q, m=sp.m, weight=sp.weight)


def _partners(modes):
    """For each mode, (index of its Phi partner (-m,n), is_conjugate).  The
    partner of a half-space mode is often the conjugate of its (m,-n) cousin."""
    key = {(md.m, md.n): i for i, md in enumerate(modes)}
    out = []
    for md in modes:
        if (-md.m, md.n) in key:
            out.append((key[(-md.m, md.n)], False))
        elif (md.m, -md.n) in key:
            out.append((key[(md.m, -md.n)], True))
        else:
            raise ValueError(f'symmetric mode needs the mode set closed under '
                             f'(m,n)->(-m,n); missing partner of ({md.m},{md.n})')
    return out


def _phi_image_field(F, S, nx, conj):
    G = np.conj(F) if conj else F
    return S * np.roll(G, nx // 2, axis=-1)


def symmetrize_fields(modes, nx):
    """Project (E_T, B) of every mode onto the Phi-symmetric subspace."""
    P = _partners(modes)
    old = [(md.ET.copy(), md.B.copy()) for md in modes]
    for i, md in enumerate(modes):
        j, cj = P[i]
        md.ET = 0.5 * (old[i][0] + _phi_image_field(old[j][0], _S_E, nx, cj))
        md.B = 0.5 * (old[i][1] + _phi_image_field(old[j][1], _S_B, nx, cj))


def symmetry_error(modes, nx) -> float:
    """max over modes of ||F - Phi(F)|| / ||F|| for (E_T, B)."""
    P = _partners(modes)
    worst = 0.0
    for i, md in enumerate(modes):
        j, cj = P[i]
        for F, Fp, S in ((md.ET, modes[j].ET, _S_E), (md.B, modes[j].B, _S_B)):
            nrm = np.linalg.norm(F)
            if nrm > 0:
                worst = max(worst, np.linalg.norm(F - _phi_image_field(Fp, S, nx, cj)) / nrm)
    return float(worst)


# ---------------------------------------------------------------------------
# checkpoint / restart and snapshot helpers
# ---------------------------------------------------------------------------
_HKEYS = ('t', 'W_K', 'W_E', 'W_B', 'W_B0', 'W_total', 'K_ion', 'K_ele', 'T_ion', 'T_ele',
          'amp', 'bx_s1', 'bx_s2', 'mode_E', 'mode_B', 'mode_E_all', 'mode_B_all')
_PKEYS = ('x', 'y', 'z', 'vx', 'vy', 'vz')
_SNAP_KEYS = ('snap_t', 'snap_Bx', 'snap_ET', 'snap_B', 'snap_prof')


def cfg_from_json(s: str) -> SimulationConfig:
    d = json.loads(s)
    d['modes'] = [ModeSpec(**m) for m in d['modes']]
    return SimulationConfig(**d)


def _cfg_core(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ('n_steps', 'diag_every')}


def _moments(species, cfg: SimulationConfig, symmetric: bool) -> np.ndarray:
    """x-binned particle moments on the full domain, shape (n_species, 5, nx):
    [count, sum vx, sum vy, sum vz, sum v^2] per x cell (images included when
    symmetric).  n(x) = count * weight / (dx * Ly * Lz); flow and temperature follow."""
    nx = cfg.nx
    dx = cfg.Lx / nx
    out = np.zeros((len(species), 5, nx))
    for isp, sp in enumerate(species):
        parts = [(sp.x, sp.vx, sp.vy, sp.vz)]
        if symmetric:
            parts.append((np.mod(sp.x + 0.5 * cfg.Lx, cfg.Lx), sp.vx, -sp.vy, sp.vz))
        for x, vx, vy, vz in parts:
            b = np.floor(x / dx).astype(np.int64) % nx
            out[isp, 0] += np.bincount(b, minlength=nx)
            out[isp, 1] += np.bincount(b, weights=vx, minlength=nx)
            out[isp, 2] += np.bincount(b, weights=vy, minlength=nx)
            out[isp, 3] += np.bincount(b, weights=vz, minlength=nx)
            out[isp, 4] += np.bincount(b, weights=vx * vx + vy * vy + vz * vz, minlength=nx)
    return out


def save_checkpoint(path, cfg, flags, dt, step, species, modes, hist, snaps):
    """Write the complete restartable state (+ all per-step history so far) atomically."""
    meta = dict(cfg=asdict(cfg), flags=flags, dt=float(dt), step=int(step),
                species=[dict(name=sp.species, q=float(sp.q), m=float(sp.m),
                              weight=float(sp.weight)) for sp in species],
                modes=[[int(md.m), int(md.n), bool(md.is_mean)] for md in modes])
    arrays = {'meta': np.array(json.dumps(meta))}
    for i, sp in enumerate(species):
        for a in _PKEYS:
            arrays[f'sp{i}_{a}'] = getattr(sp, a)
    for j, md in enumerate(modes):
        arrays[f'mode{j}_ET'] = md.ET
        arrays[f'mode{j}_B'] = md.B
    for k in _HKEYS:
        arrays[f'h_{k}'] = hist[k][:step]
    for k in _SNAP_KEYS:
        arrays[k] = np.array(snaps[k])
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path) -> dict:
    """Read a checkpoint written by save_checkpoint.  Returns dict with keys
    meta, cfg, species (list of dicts of arrays), modes (list of (ET, B)),
    hist (dict of arrays), snaps (dict of arrays)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    sp = [{a: z[f'sp{i}_{a}'] for a in _PKEYS} for i in range(len(meta['species']))]
    modes = [(z[f'mode{j}_ET'], z[f'mode{j}_B']) for j in range(len(meta['modes']))]
    hist = {k: z[f'h_{k}'] for k in _HKEYS}
    snaps = {k: z[k] for k in _SNAP_KEYS}
    cfg = cfg_from_json(json.dumps(meta['cfg']))
    return dict(meta=meta, cfg=cfg, species=sp, modes=modes, hist=hist, snaps=snaps)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def run_coupled_loop(cfg: SimulationConfig, n_steps: int | None = None,
                     dt: float | None = None, seed: int = 12345,
                     evolve_mean: bool = True, log_every: int = 0,
                     paired_load: bool = False, symmetric: bool = False,
                     snapshots: int = 0, snapshot_every: int = 0,
                     checkpoint_prefix: str | None = None, checkpoint_every: int = 0,
                     keep_every: int = 0, resume_from: str | None = None):
    """Run the loop.  Returns a dict of per-step arrays:
      t, W_K, W_E, W_B (perturbation), W_B0 (mean-field magnetic, only if
      evolve_mean), W_total, K_ion, K_ele, T_ion, T_ele (thermal energy per
      particle), amp (n_steps x n_pert_modes, max|Bx_k|), plus dt, modes, species.

    evolve_mean=True  : the (0,0) mode (E_0, B_0 = B_eq + dB) is evolved from the
                        particle mean current, so the equilibrium magnetic energy is
                        included in the physical evolution.
    evolve_mean=False : B_eq is held static as a fixed external field
                        (perturbation-only energy accounting).

    The mean mode is the default physical behavior for these Harris-sheet runs,
    so the flag defaults to True for backward compatibility with the intended
    equilibrium evolution.  Passing False is only for intentionally static-
    background tests.

    Snapshots (every snapshot_every steps, or n_steps//snapshots): Bx_k(x), all-mode
    (E_T, B) fields, and x-binned particle moments -> snap_* arrays.

    Checkpoints (checkpoint_prefix + checkpoint_every): the complete restartable state
    and all history so far is written to <prefix>_ckpt_latest.npz every
    checkpoint_every steps (plus <prefix>_ckpt_<step>.npz every keep_every steps and
    <prefix>_final.npz at the end).  resume_from=<ckpt> continues exactly from there
    (n_steps is the TOTAL target step count).
    """
    n_steps = cfg.n_steps if n_steps is None else n_steps
    units = derive_cfg_units(cfg)
    dx = cfg.Lx / cfg.nx
    vperp = cfg.Ly * cfg.Lz
    eps_0, mu_0, c2 = units.eps_0, units.mu_0, units.c ** 2
    dt = stable_dt(cfg, units) if dt is None else dt

    ck = None
    start_step = 0
    if resume_from:
        ck = load_checkpoint(resume_from)
        fl = ck['meta']['flags']
        if (bool(fl['evolve_mean']), bool(fl['symmetric'])) != (evolve_mean, symmetric):
            raise ValueError('resume: evolve_mean/symmetric flags differ from the checkpoint')
        if _cfg_core(ck['meta']['cfg']) != _cfg_core(asdict(cfg)):
            raise ValueError('resume: configuration differs from the checkpoint')
        dt = ck['meta']['dt']
        start_step = ck['meta']['step']
    if symmetric and cfg.nx % 2:
        raise ValueError('symmetric mode needs an even nx')
    kfac = 2.0 if symmetric else 1.0     # real + image particles carry equal energy
    if ck is None:
        rng = np.random.default_rng(seed)
        species = [load_species(cfg, units, 'ion', rng, half_domain=symmetric),
                   load_species(cfg, units, 'electron', rng, half_domain=symmetric)]
        if paired_load:
            # quiet start: electrons at the ion positions => rho(t=0) = 0 exactly,
            # so no spurious shot-noise field energy relaxes into particle heat.
            species[1].x = species[0].x.copy()
            species[1].y = species[0].y.copy()
            species[1].z = species[0].z.copy()
    else:
        species = [ParticleSet(species=m['name'], weight=m['weight'], q=m['q'], m=m['m'],
                               **{a: arrs[a].copy() for a in _PKEYS})
                   for m, arrs in zip(ck['meta']['species'], ck['species'])]
    modes = build_modes(cfg, include_mean=evolve_mean)
    if ck is None:
        if symmetric:
            symmetrize_fields(modes, cfg.nx)
    else:
        if [[md.m, md.n, md.is_mean] for md in modes] != ck['meta']['modes']:
            raise ValueError('resume: mode set differs from the checkpoint')
        for md, (et, bb) in zip(modes, ck['modes']):
            md.ET, md.B = et.copy(), bb.copy()
    mults = [md.mult for md in modes]
    n_pert = sum(1 for md in modes if not md.is_mean)

    keys = ('t', 'W_K', 'W_E', 'W_B', 'W_B0', 'W_total',
            'W_B_full', 'W_total_full', 'K_ion', 'K_ele', 'T_ion', 'T_ele')
    hist = {k: np.zeros(n_steps) for k in keys}
    hist['amp'] = np.zeros((n_steps, n_pert))
    hist['bx_s1'] = np.zeros((n_steps, n_pert))   # |Bx_k| at x = Lx/4
    hist['bx_s2'] = np.zeros((n_steps, n_pert))   # |Bx_k| at x = 3Lx/4
    hist['mode_E'] = np.zeros((n_steps, n_pert))  # electric energy per non-mean perturbation mode
    hist['mode_B'] = np.zeros((n_steps, n_pert))  # magnetic energy per non-mean perturbation mode
    hist['mode_E_all'] = np.zeros((n_steps, len(modes)))
    hist['mode_B_all'] = np.zeros((n_steps, len(modes)))
    W_B_bg = _background_magnetic_energy(cfg, mu_0) if not evolve_mean else 0.0
    i_s1, i_s2 = cfg.nx // 4, (3 * cfg.nx) // 4
    snap_every = snapshot_every or (max(1, n_steps // snapshots) if snapshots else 0)
    snap = {k: [] for k in _SNAP_KEYS}
    flags = dict(evolve_mean=evolve_mean, symmetric=symmetric, paired_load=paired_load, seed=seed)
    if ck is not None:
        if n_steps < start_step:
            raise ValueError('resume: n_steps (total) is smaller than the checkpoint step')
        for k in _HKEYS:
            hist[k][:start_step] = ck['hist'][k]
        for k in _SNAP_KEYS:
            snap[k] = list(ck['snaps'][k])

    for step in range(start_step, n_steps):
        # positions x^n: CIC weights and mode phases (shared by deposit + gather)
        shapes = [_cic(sp.x, dx, cfg.nx) for sp in species]
        phases = [[_phase(md, sp.y, sp.z) for md in modes] for sp in species]

        # phi^n from rho^n (U_L = 1/2 int rho phi); B^{n+1/2}; B^n
        if symmetric:
            imgs = [_image(sp, cfg) for sp in species]
            dep_sp = species + imgs
            dep_shapes = shapes + [_cic(im.x, dx, cfg.nx) for im in imgs]
            dep_phases = phases + [[_phase(md, im.y, im.z) for md in modes] for im in imgs]
        else:
            dep_sp, dep_shapes, dep_phases = species, shapes, phases
        rho = deposit_rho(dep_sp, cfg, modes, dx, dep_shapes, dep_phases)
        phis, B_new, B_n = [], [], []
        W_E = 0.0
        ipe = 0
        for km, md in enumerate(modes):
            phi = solve_potential(rho[km], md, eps_0)
            phis.append(phi)
            we = (0.5 * md.mult * dx * float(np.sum((np.conj(rho[km]) * phi).real))
                  + _mode_energy(md.ET, dx, vperp, 0.5 * eps_0, mult=md.mult, mean_mode=md.is_mean))
            W_E += we
            hist['mode_E_all'][step, km] = we
            if not md.is_mean:
                hist['mode_E'][step, ipe] = we
                ipe += 1
            curlE = np.array(curl_mode(md.ET[0], md.ET[1], md.ET[2], md.ky, md.kz, dx))
            Bnew = md.B - dt * curlE
            B_new.append(Bnew)
            B_n.append(0.5 * (md.B + Bnew))
        E_T = [md.ET for md in modes]

        # Boris push: E from modes, B = B_eq + B_pert
        W_K = 0.0
        Ks, Ts = [], []
        mid = []
        for sp, shape, ph in zip(species, shapes, phases):
            Ep = gather_longitudinal(modes, phis, shape, ph, dx) + gather(E_T, shape, ph, mults)
            Bp = gather(B_n, shape, ph, mults)
            if not evolve_mean:
                Bp[1] += cfg.Bg
                Bp[2] += Bz_eq(sp.x, cfg)
            v = np.vstack([sp.vx, sp.vy, sp.vz])
            v_new = boris(v, Ep, Bp, sp.q / sp.m, dt)
            Ks.append(kfac * 0.5 * sp.m * sp.weight * float(np.sum(v * v_new)))
            Ts.append(_temperature(sp, v_new, cfg))
            W_K += Ks[-1]
            mid.append((sp.x + 0.5 * dt * v_new[0],
                        sp.y + 0.5 * dt * v_new[1],
                        sp.z + 0.5 * dt * v_new[2], v_new))
            sp.x = np.mod(sp.x + dt * v_new[0], cfg.Lx)
            sp.y = np.mod(sp.y + dt * v_new[1], cfg.Ly)
            sp.z = np.mod(sp.z + dt * v_new[2], cfg.Lz)
            sp.vx, sp.vy, sp.vz = v_new

        # J^{n+1/2} at the mid-step position, then enforce the discrete continuity
        # relation before projecting to the transverse current.
        J = [np.zeros((3, cfg.nx), dtype=np.complex128) for _ in modes]
        jsrc = [(sp, xm, ym, zm, v_new) for sp, (xm, ym, zm, v_new) in zip(species, mid)]
        if symmetric:
            jsrc += [(sp, xm + 0.5 * cfg.Lx, -ym, zm, v_new * _VY_FLIP)
                     for sp, (xm, ym, zm, v_new) in zip(species, mid)]
        for sp, xm, ym, zm, v_new in jsrc:
            i0, i1, f = _cic(xm, dx, cfg.nx)
            coef = sp.q * sp.weight / dx
            for km, md in enumerate(modes):
                cph = coef * np.conj(_phase(md, ym, zm))
                for a in range(3):
                    J[km][a] += _deposit(i0, i1, f, cph * v_new[a], cfg.nx)

        W_B = 0.0
        W_B0 = 0.0
        ip = 0
        for km, md in enumerate(modes):
            wb = _mode_energy(md.B, dx, vperp, 0.5 / mu_0, b=B_new[km], mult=md.mult,
                              mean_mode=md.is_mean)
            hist['mode_B_all'][step, km] = wb
            if md.is_mean:
                W_B0 += wb
            else:
                W_B += wb
                hist['mode_B'][step, ip] = wb
                hist['amp'][step, ip] = float(np.max(np.abs(B_new[km][0])))
                hist['bx_s1'][step, ip] = float(np.abs(B_new[km][0][i_s1]))
                hist['bx_s2'][step, ip] = float(np.abs(B_new[km][0][i_s2]))
                ip += 1
            JT = transverse_current(J[km], md, dx)
            curlB = np.array(curl_mode(B_new[km][0], B_new[km][1], B_new[km][2],
                                       md.ky, md.kz, dx))
            md.ET = md.ET + dt * (c2 * curlB - JT / eps_0)
            md.B = B_new[km]

        if snap_every and (step % snap_every == 0 or step == n_steps - 1):
            snap['snap_t'].append(step * dt)
            snap['snap_Bx'].append(np.array([md.B[0].copy() for md in modes if not md.is_mean]))
            snap['snap_ET'].append(np.array([md.ET.copy() for md in modes]))
            snap['snap_B'].append(np.array([md.B.copy() for md in modes]))
            snap['snap_prof'].append(_moments(species, cfg, symmetric))

        W_B_full = W_B + W_B0 + W_B_bg if not evolve_mean else W_B + W_B0
        hist['t'][step] = step * dt
        hist['W_K'][step] = W_K
        hist['W_E'][step] = W_E
        hist['W_B'][step] = W_B
        hist['W_B0'][step] = W_B0
        hist['W_total'][step] = W_K + W_E + W_B + W_B0
        hist['W_B_full'][step] = W_B_full
        hist['W_total_full'][step] = W_K + W_E + W_B_full
        hist['K_ion'][step], hist['K_ele'][step] = Ks
        hist['T_ion'][step], hist['T_ele'][step] = Ts
        if log_every and step % log_every == 0:
            w0 = hist['W_total'][0]
            print(f'step {step:6d} t={step * dt:8.2f} W_K={W_K:.5e} W_B0={W_B0:.5e} '
                  f'W_E={W_E:.3e} W_Bpert={W_B:.3e} dW/W={(hist["W_total"][step] - w0) / w0:+.2e}',
                  flush=True)

        if checkpoint_prefix and checkpoint_every and (
                (step + 1) % checkpoint_every == 0 or step == n_steps - 1):
            latest = f'{checkpoint_prefix}_ckpt_latest.npz'
            save_checkpoint(latest, cfg, flags, dt, step + 1, species, modes, hist, snap)
            if keep_every and (step + 1) % keep_every == 0:
                shutil.copyfile(latest, f'{checkpoint_prefix}_ckpt_{step + 1:07d}.npz')
            if step == n_steps - 1:
                shutil.copyfile(latest, f'{checkpoint_prefix}_final.npz')

    hist['dt'] = dt
    for k in _SNAP_KEYS:
        hist[k] = np.array(snap[k])   # snap_Bx: (n_snap, n_pert, nx) complex, Bx_k(x)
    hist['sym_err'] = symmetry_error(modes, cfg.nx) if symmetric else 0.0
    hist['modes'] = modes
    hist['species'] = species
    return hist


def run_short_mode_loop(cfg: SimulationConfig, n_steps: int = 10, diag_every: int = 1):
    """Backward-compatible wrapper: (total_energy_history, modes, species)."""
    h = run_coupled_loop(cfg, n_steps=n_steps)
    return h['W_total'], h['modes'], h['species']


__all__ = [
    'canonical_modes',
    'build_modes',
    'stable_dt',
    'run_coupled_loop',
    'run_short_mode_loop',
]
