"""GPU-resident direct spectral 3D EM/PIC solver (no x-grid, no FFT).

Same scheme as the legacy CPU code (coupled_loop.py), but spectral in all three dimensions:

    E = E_L + E_T        E_L: curl-free, from Poisson on rho^n      (phi_k = rho_k / (eps0 k^2), E_L = -i k phi)
                         E_T: divergence-free, evolved by Ampere with the TRANSVERSE current J_T
    time levels          x^n, v^{n-1/2}, E_T^n, B^{n-1/2}

    1. rho^n   = deposit(x^n)                                                        [kernel, 1 quantity]
       E_L^n   = Poisson(rho^n)                                                      [O(K)]
    2. B^{n+1/2} = B^{n-1/2} - dt curl E_T^n ;   B^n = (B^{n-1/2} + B^{n+1/2}) / 2    [O(K)] (Faraday)
    3. gather E_L + E_T and B^n at x^n, Boris push -> v^{n+1/2}, x^{n+1}               [kernel]
    4. J^{n+1/2} = deposit(x^{n+1/2} = x^{n+1} - dt/2 v^{n+1/2}, v^{n+1/2})            [kernel, 3 quantities]
       J_T = J - k (k.J) / k^2                                                       [O(K)]
    5. E_T^{n+1} = E_T^n + dt (c^2 curl B^{n+1/2} - J_T / eps0)                       [O(K)] (Ampere)

Energies at time n use the legacy time-centred forms:
    W_K = sum 1/2 m W v^{n-1/2}.v^{n+1/2},  W_B = 1/(2 mu0) B^{n-1/2}.B^{n+1/2},
    W_E = 1/2 eps0 (|E_L|^2 + |E_T|^2) integrated (E_L and E_T are orthogonal).

Conventions
-----------
* f(x) = sum_k f_k e^{i k.x}; rho_k = (1/V) sum_p q W_p e^{-i k.x_p} (W_p = physical particles per macro-particle).
* Only m_x >= 0 is stored (Hermitian symmetry f_{-k} = conj f_k): shapes (..., A, B, C) with
  A = hx + 1, B = 2 hy + 1, C = 2 hz + 1; complex numbers are separate re/im planes.
* eps0, c are parameters, mu0 = 1/(eps0 c^2).  B_eq is a uniform external background field.
* Every O(K) field operation is a compiled MLX graph on the GPU; there is no Python loop over modes
  or particles, and no host transfer inside a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .spectral_kernels import DepositPlan, deposit, gather_push


@dataclass
class PICConfig:
    L: tuple[float, float, float] = (256.0, 256.0, 256.0)
    modes: tuple[int, int, int] = (3, 1, 1)  # hx, hy, hz
    dt: float = 0.05
    c: float = 1.0
    eps0: float = 1.0
    B_eq: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def shape(self) -> tuple[int, int, int]:
        hx, hy, hz = self.modes
        return hx + 1, 2 * hy + 1, 2 * hz + 1

    @property
    def volume(self) -> float:
        return self.L[0] * self.L[1] * self.L[2]

    @property
    def mu0(self) -> float:
        return 1.0 / (self.eps0 * self.c**2)


@dataclass
class Species:
    q: float
    m: float
    weight: float  # physical particles per macro-particle
    state: tuple = field(default=None)  # six float32 mx arrays (x, y, z, vx, vy, vz)
    plan_rho: DepositPlan = None
    plan_J: DepositPlan = None
    gather_params: mx.array = None
    rho_params: mx.array = None
    J_params: mx.array = None

    @property
    def n(self) -> int:
        return self.state[0].shape[0]


def mode_vectors(cfg: PICConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wave vectors on the half-space mode grid, each of shape (A, B, C)."""
    A, B, C = cfg.shape
    hy, hz = cfg.modes[1], cfg.modes[2]
    kx = 2 * np.pi * np.arange(A) / cfg.L[0]
    ky = 2 * np.pi * (np.arange(B) - hy) / cfg.L[1]
    kz = 2 * np.pi * (np.arange(C) - hz) / cfg.L[2]
    return tuple(np.broadcast_to(v, (A, B, C)).astype(np.float32).copy() for v in np.meshgrid(kx, ky, kz, indexing="ij"))


def fold_weights(shape) -> np.ndarray:
    """1 on the m_x = 0 plane (both +-(b,c) are stored), 2 for m_x > 0 (the -k partner is implied)."""
    A, _, _ = shape
    w = np.full((A, 1, 1), 2.0, dtype=np.float32)
    w[0] = 1.0
    return w


def hermitian_project(F: np.ndarray) -> np.ndarray:
    """Make a complex (..., A, B, C) array consistent with a real field: on the m_x = 0 plane
    F[0, b, c] = conj(F[0, -b, -c]); the mean mode becomes real."""
    F = F.copy()
    plane = F[..., 0, :, :]
    F[..., 0, :, :] = 0.5 * (plane + np.conj(plane[..., ::-1, ::-1]))
    return F


class SpectralPIC:
    def __init__(self, cfg: PICConfig, species: list[Species], E_T0: np.ndarray | None = None,
                 B0: np.ndarray | None = None) -> None:
        mx.set_default_device(mx.gpu)
        self.cfg = cfg
        self.species = species
        self.shape = cfg.shape
        A, B, C = self.shape
        kx, ky, kz = mode_vectors(cfg)
        self.k = mx.array(np.stack([kx, ky, kz]))  # (3, A, B, C)
        k2 = kx**2 + ky**2 + kz**2
        self._k2_np = k2.astype(np.float64)
        self.inv_k2 = mx.array(np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0).astype(np.float32))
        self._fold = mx.array(fold_weights(self.shape))
        self.c2 = cfg.c**2
        self.step_count = 0
        self.diag: list[mx.array] = []

        zeros = np.zeros((3, A, B, C), dtype=np.complex128)
        self.ET = self._pack(self._transverse(hermitian_project(zeros if E_T0 is None else E_T0)))
        self.Bh = self._pack(self._transverse(hermitian_project(zeros if B0 is None else B0)))  # B^{n-1/2}

        k0 = (2 * np.pi / cfg.L[0], 2 * np.pi / cfg.L[1], 2 * np.pi / cfg.L[2])
        for sp in species:
            sp.plan_rho = DepositPlan(self.shape, sp.n, nq=1)
            sp.plan_J = DepositPlan(self.shape, sp.n, nq=3)
            h = sp.q / sp.m * cfg.dt / 2
            qw = sp.q * sp.weight / cfg.volume
            sp.gather_params = mx.array(np.array([h, cfg.dt, *cfg.L, *k0, *cfg.B_eq], dtype=np.float32))
            sp.rho_params = mx.array(np.array([qw, *k0, 0.0], dtype=np.float32))
            sp.J_params = mx.array(np.array([qw, *k0, cfg.dt / 2], dtype=np.float32))

        self._faraday = mx.compile(self._faraday_impl)
        self._fields = mx.compile(self._fields_impl)
        self._ampere = mx.compile(self._ampere_impl)
        mx.eval(self.ET, self.Bh, *[a for sp in species for a in sp.state])

    # ---------------------------------------------------------------- packing / initial fields
    def _transverse(self, Fc: np.ndarray) -> np.ndarray:
        """Remove the longitudinal part of a complex (3, A, B, C) field (float64, host, setup only)."""
        k = np.stack([np.asarray(a, dtype=np.float64) for a in self.k])
        k2 = np.where(self._k2_np > 0, self._k2_np, 1.0)
        return Fc - k * (np.sum(k * Fc, axis=0) / k2)

    @staticmethod
    def _pack(Fc: np.ndarray) -> mx.array:
        return mx.array(np.stack([Fc.real, Fc.imag], axis=1).astype(np.float32))  # (3, 2, A, B, C)

    @staticmethod
    def unpack(F: mx.array) -> np.ndarray:
        a = np.asarray(F, dtype=np.float64)
        return a[:, 0] + 1j * a[:, 1]

    # ---------------------------------------------------------------- O(K) field math (compiled, on GPU)
    def _curl(self, F):
        """i k x F for F of shape (3, 2, A, B, C) (re/im planes)."""
        kx, ky, kz = self.k[0], self.k[1], self.k[2]

        def kcross(V):
            return mx.stack([ky * V[2] - kz * V[1], kz * V[0] - kx * V[2], kx * V[1] - ky * V[0]])

        return mx.stack([-kcross(F[:, 1]), kcross(F[:, 0])], axis=1)

    def _faraday_impl(self, Bh, ET, dt):
        Bnext = Bh - dt * self._curl(ET)
        return Bnext, 0.5 * (Bh + Bnext)

    def _fields_impl(self, rho, ET, Bn):
        """rho (2, A, B, C) -> Poisson E_L; returns the folded gather table
        [a][b][c][comp(E0,E1,E2,B0,B1,B2)][re,im] and the longitudinal energy sum."""
        s = rho * (self.inv_k2 / self.cfg.eps0)  # (2, A, B, C): phi_k
        # E_L = -i k phi:  re = k * phi_im,  im = -k * phi_re
        EL = mx.stack([self.k * s[1], -self.k * s[0]], axis=1)  # (3, 2, A, B, C)
        F = mx.concatenate([EL + ET, Bn], axis=0)  # (6, 2, A, B, C)
        F = mx.transpose(F, (2, 3, 4, 0, 1))  # (A, B, C, 6, 2)
        table = F * self._fold.reshape(-1, 1, 1, 1, 1)
        wEL = mx.sum(self._fold * mx.sum(EL * EL, axis=(0, 1)))
        return table, wEL

    def _ampere_impl(self, ET, Bnext, J, dt):
        """E_T^{n+1} = E_T^n + dt (c^2 curl B^{n+1/2} - J_T / eps0), J_T = J - k (k.J) / k^2."""
        kJ = mx.sum(self.k[:, None] * J, axis=0)  # (2, A, B, C)
        JT = J - self.k[:, None] * (kJ * self.inv_k2)[None]
        return ET + dt * (self.c2 * self._curl(Bnext) - JT / self.cfg.eps0)

    # ---------------------------------------------------------------- one time step
    def step(self, diag: bool = False) -> None:
        cfg, dt = self.cfg, self.cfg.dt
        A, B, C = self.shape

        rho = None
        for sp in self.species:
            r = deposit(sp.state, sp.rho_params, sp.plan_rho)  # (2, A, B, C)
            rho = r if rho is None else rho + r
        Bnext, Bn = self._faraday(self.Bh, self.ET, dt)
        table, wEL = self._fields(rho, self.ET, Bn)

        old_v = [sp.state[3:] for sp in self.species]
        J = None
        for sp in self.species:
            sp.state = gather_push(sp.state, table, sp.gather_params, self.shape)
            j = deposit(sp.state, sp.J_params, sp.plan_J)  # (6, A, B, C)
            J = j if J is None else J + j
        J = J.reshape(3, 2, A, B, C)

        if diag:
            wk = 0.0
            for sp, vo in zip(self.species, old_v):
                wk = wk + 0.5 * sp.m * sp.weight * sum(mx.sum(a * b) for a, b in zip(vo, sp.state[3:]))
            fold = self._fold.reshape(-1, 1, 1)
            wET = mx.sum(fold * mx.sum(self.ET * self.ET, axis=(0, 1)))
            wB = mx.sum(fold * mx.sum(self.Bh * Bnext, axis=(0, 1)))
            V = cfg.volume
            self.diag.append(mx.stack([wk, 0.5 * cfg.eps0 * V * (wEL + wET), 0.5 / cfg.mu0 * V * wB]))
            self._rho_last = rho

        self.ET = self._ampere(self.ET, Bnext, J, dt)
        self.Bh = Bnext
        self.step_count += 1

    def run(self, steps: int, eval_every: int = 1, diag_every: int = 0) -> None:
        for i in range(steps):
            self.step(diag=bool(diag_every) and (i % diag_every == 0))
            if (i + 1) % eval_every == 0:
                self.sync()
        self.sync()

    def sync(self) -> None:
        mx.eval(self.ET, self.Bh, *self.diag[-1:], *[a for sp in self.species for a in sp.state])

    # ---------------------------------------------------------------- diagnostics
    def energy_history(self) -> np.ndarray:
        """(n_diag, 3) array of [W_K, W_E, W_B] (time-centred, legacy definitions)."""
        if not self.diag:
            return np.zeros((0, 3))
        out = mx.stack(self.diag)
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)
