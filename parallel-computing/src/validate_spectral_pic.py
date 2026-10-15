"""Validate the GPU spectral PIC (Poisson + Faraday + Ampere + Boris) three ways.

1. Against an independent float64 NumPy implementation of the SAME scheme: full mode set (both signs of
   m_x), direct complex exponentials, explicit Poisson / projection -- no shared code or shortcuts.
2. Gauss's law: i k.E = rho/eps0 must hold to round-off at every step (it is imposed by Poisson).
3. Physics: a Langmuir oscillation (cold electrons, sinusoidal displacement) must ring at omega_pe,
   and a neutral thermal plasma must conserve total energy.

    PYTHONPATH=. python -m src.validate_spectral_pic
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from .spectral_pic_gpu import PICConfig, Species, SpectralPIC


def full_modes(cfg: PICConfig):
    hx, hy, hz = cfg.modes
    m = np.array([(a, b, c) for a in range(-hx, hx + 1) for b in range(-hy, hy + 1) for c in range(-hz, hz + 1)])
    return m, 2 * np.pi * m / np.array(cfg.L)


def random_hermitian(m: np.ndarray, rng, amp: float, ncomp: int = 3) -> np.ndarray:
    """Complex amplitudes F[comp, mode] with F(-k) = conj F(k)."""
    R = amp * (rng.normal(size=(ncomp, len(m))) + 1j * rng.normal(size=(ncomp, len(m))))
    idx = {tuple(x): i for i, x in enumerate(m)}
    neg = np.array([idx[tuple(-x)] for x in m])
    return 0.5 * (R + np.conj(R[:, neg]))


def transverse(F: np.ndarray, k: np.ndarray) -> np.ndarray:
    """F (3, K), k (K, 3): remove the longitudinal part."""
    k2 = np.sum(k**2, axis=1)
    k2[k2 == 0] = 1.0
    return F - k.T * (np.sum(k * F.T, axis=1) / k2)


def to_half(F_full: np.ndarray, m: np.ndarray, cfg: PICConfig) -> np.ndarray:
    A, B, C = cfg.shape
    out = np.zeros((F_full.shape[0], A, B, C), dtype=np.complex128)
    hy, hz = cfg.modes[1], cfg.modes[2]
    for i, (a, b, c) in enumerate(m):
        if a >= 0:
            out[:, a, b + hy, c + hz] = F_full[:, i]
    return out


class Reference:
    """float64 NumPy leapfrog spectral PIC on the full mode set (Poisson + transverse Ampere)."""

    def __init__(self, cfg, species_state, species_meta, ET, B):
        self.cfg = cfg
        self.m, self.k = full_modes(cfg)
        self.k2 = np.sum(self.k**2, axis=1)
        self.inv_k2 = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1.0), 0.0)
        self.sp = [dict(zip("x y z vx vy vz".split(), [a.astype(np.float64) for a in st]), q=q, mass=mm, w=w)
                   for st, (q, mm, w) in zip(species_state, species_meta)]
        self.ET = ET.copy()
        self.Bh = B.copy()
        self.L = np.array(cfg.L)

    def curl(self, F):
        return 1j * np.cross(self.k, F.T).T

    def step(self):
        cfg, dt = self.cfg, self.cfg.dt
        V = cfg.volume
        # --- rho^n at x^n, Poisson
        rho = np.zeros(len(self.k), dtype=np.complex128)
        for s in self.sp:
            pos = np.stack([s["x"], s["y"], s["z"]], axis=1)
            rho += s["q"] * s["w"] / V * np.exp(-1j * pos @ self.k.T).sum(axis=0)
        phi = rho * self.inv_k2 / cfg.eps0
        EL = -1j * self.k.T * phi
        self.rho, self.EL_last = rho, EL
        # --- Faraday
        Bnext = self.Bh - dt * self.curl(self.ET)
        Bn = 0.5 * (self.Bh + Bnext)
        # --- gather + Boris + J deposit at the mid-step
        J = np.zeros_like(self.ET)
        Etot = EL + self.ET
        for s in self.sp:
            pos = np.stack([s["x"], s["y"], s["z"]], axis=1)
            v = np.stack([s["vx"], s["vy"], s["vz"]], axis=1)
            ph = np.exp(1j * pos @ self.k.T)
            E = (ph @ Etot.T).real
            B = (ph @ Bn.T).real + np.array(cfg.B_eq)
            h = s["q"] / s["mass"] * dt / 2
            vm = v + h * E
            t = h * B
            ss = 2 * t / (1 + np.sum(t * t, axis=1, keepdims=True))
            vp = vm + np.cross(vm, t)
            vn = vm + np.cross(vp, ss) + h * E
            pos = np.mod(pos + dt * vn, self.L)
            s["x"], s["y"], s["z"] = pos.T
            s["vx"], s["vy"], s["vz"] = vn.T
            mid = pos - 0.5 * dt * vn
            J += s["q"] * s["w"] / V * (np.exp(-1j * mid @ self.k.T).T @ vn).T
        JT = transverse(J, self.k)
        self.ET = self.ET + dt * (cfg.c**2 * self.curl(Bnext) - JT / cfg.eps0)
        self.Bh = Bnext


def per_dist(a, b, L):
    d = np.abs(a - b)
    return np.minimum(d, L - d)


def compare_to_reference(args) -> None:
    cfg = PICConfig(L=(40.0, 30.0, 20.0), modes=tuple(args.modes), dt=0.05, B_eq=(0.2, -0.1, 0.5))
    rng = np.random.default_rng(1)
    m, k = full_modes(cfg)
    ETf = transverse(random_hermitian(m, rng, args.amp), k)
    Bf = transverse(random_hermitian(m, rng, args.amp), k)
    meta = [(+1.0, 1.0, 2.0), (-1.0, 0.1, 2.0)]
    states = []
    for q, mm, w in meta:
        pos = rng.uniform(0, 1, (args.n, 3)) * np.array(cfg.L)
        vel = rng.normal(0, 0.2, (args.n, 3))
        states.append([pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2]])
    species = []
    for (q, mm, w), st in zip(meta, states):
        sp = Species(q=q, m=mm, weight=w)
        sp.state = tuple(mx.array(a.astype(np.float32)) for a in st)
        species.append(sp)
    gpu = SpectralPIC(cfg, species, E_T0=to_half(ETf, m, cfg), B0=to_half(Bf, m, cfg))
    ref = Reference(cfg, [[a.astype(np.float32).astype(np.float64) for a in st] for st in states], meta, ETf, Bf)

    A, B, C = cfg.shape
    print(f"[1] vs float64 NumPy: modes {cfg.modes}: {A*B*C} stored (half space), {len(m)} in the full reference set, "
          f"N={args.n}/species, field amp={args.amp}")
    print(f"{'step':>4} {'max|dx|':>10} {'max|dv|':>10} {'max|dE_T|/max|E_T|':>19} {'max|dB|/max|B|':>15}")
    for step in range(1, args.steps + 1):
        gpu.run(1, diag_every=1)
        ref.step()
        dx = max(max(per_dist(np.asarray(sp.state[i], dtype=np.float64), ref.sp[j][n], cfg.L[i]).max() for i, n in enumerate("xyz"))
                 for j, sp in enumerate(gpu.species))
        dv = max(max(np.abs(np.asarray(sp.state[3 + i], dtype=np.float64) - ref.sp[j][n]).max() for i, n in enumerate(("vx", "vy", "vz")))
                 for j, sp in enumerate(gpu.species))
        Eh, Bh = SpectralPIC.unpack(gpu.ET), SpectralPIC.unpack(gpu.Bh)
        Er, Br = to_half(ref.ET, m, cfg), to_half(ref.Bh, m, cfg)
        print(f"{step:>4} {dx:>10.2e} {dv:>10.2e} {np.abs(Eh - Er).max() / max(np.abs(Er).max(), 1e-30):>19.2e} "
              f"{np.abs(Bh - Br).max() / max(np.abs(Br).max(), 1e-30):>15.2e}")

    # rho^n deposited by the GPU kernel vs the float64 direct sum (same step), and E_T must stay divergence-free
    rho_g = SpectralPIC.unpack(gpu._rho_last[None])[0]
    rho_r = to_half(ref.rho[None], m, cfg)[0]
    d_rho = np.abs(rho_g - rho_r).max() / np.abs(rho_r).max()
    kk = np.asarray(gpu.k, dtype=np.float64)
    ET = SpectralPIC.unpack(gpu.ET)
    div_et = np.abs(np.sum(kk * ET, axis=0)).max() / max(np.abs(ET).max() * np.abs(kk).max(), 1e-30)
    print(f"[2] deposited rho_k: max|GPU - float64|/max|rho| = {d_rho:.1e};  E_T divergence |k.E_T|/(|k||E_T|) = {div_et:.1e}")


def langmuir(args) -> None:
    """Cold electrons displaced by delta*sin(k x) with fixed ions: E_L energy must oscillate at 2 omega_pe."""
    L = (64.0, 8.0, 8.0)
    cfg = PICConfig(L=L, modes=(2, 0, 0), dt=0.05, eps0=1.0, c=10.0, B_eq=(0.0, 0.0, 0.0))
    n = 200_000
    rng = np.random.default_rng(3)
    Wphys = 1.0
    n0 = n * Wphys / cfg.volume
    q_e, m_e = -1.0, 1.0
    wpe = np.sqrt(n0 * q_e**2 / (cfg.eps0 * m_e))
    pos = rng.uniform(0, 1, (n, 3)) * np.array(L)
    k1 = 2 * np.pi / L[0]
    delta = 0.02 / k1
    pos_e = pos.copy()
    pos_e[:, 0] = np.mod(pos[:, 0] + delta * np.sin(k1 * pos[:, 0]), L[0])
    zeros = np.zeros((3, n))
    ions = Species(q=+1.0, m=1e9, weight=Wphys)
    ions.state = tuple(mx.array(a.astype(np.float32)) for a in (*pos.T, *zeros))
    ele = Species(q=q_e, m=m_e, weight=Wphys)
    ele.state = tuple(mx.array(a.astype(np.float32)) for a in (*pos_e.T, *zeros))
    pic = SpectralPIC(cfg, [ions, ele])
    steps = int(2 * np.pi / wpe / cfg.dt * 3)
    pic.run(steps, eval_every=20, diag_every=1)
    h = pic.energy_history()
    we = h[:, 1]
    t = np.arange(len(we)) * cfg.dt
    f = np.fft.rfftfreq(len(we), cfg.dt) * 2 * np.pi
    sp_ = np.abs(np.fft.rfft(we - we.mean() )) ** 2
    pad = 8
    spec = np.abs(np.fft.rfft((we - we.mean()) * np.hanning(len(we)), n=len(we) * pad))
    fr = np.fft.rfftfreq(len(we) * pad, cfg.dt) * 2 * np.pi
    w_meas = fr[1:][np.argmax(spec[1:])] / 2.0
    tot = h.sum(axis=1)
    print(f"[3a] Langmuir: omega_pe theory = {wpe:.4f}, measured = {w_meas:.4f}  ({100 * abs(w_meas - wpe) / wpe:.2f}% off);"
          f"  total-energy variation over {steps} steps = {np.ptp(tot) / tot.mean():.2e}")


def thermal_energy(args) -> None:
    cfg = PICConfig(L=(256.0, 256.0, 256.0), modes=(3, 1, 1), dt=0.05, eps0=1.0, c=5.0, B_eq=(0.0, 0.0, 1.0))
    rng = np.random.default_rng(5)
    n = 200_000
    sps = []
    for q, m, vth in ((+1.0, 1.0, 0.05), (-1.0, 0.1, 0.15)):
        sp = Species(q=q, m=m, weight=1.0)
        pos = rng.uniform(0, 256, (3, n))
        vel = rng.normal(0, vth, (3, n))
        sp.state = tuple(mx.array(c.astype(np.float32)) for c in (*pos, *vel))
        sps.append(sp)
    pic = SpectralPIC(cfg, sps)
    pic.run(args.thermal_steps, eval_every=10, diag_every=10)
    h = pic.energy_history()
    tot = h.sum(axis=1)
    print(f"[3b] thermal plasma, {args.thermal_steps} steps: W_K={h[0, 0]:.4e} W_E {h[:, 1].min():.2e}..{h[:, 1].max():.2e}  "
          f"max |dW/W0| = {np.abs(tot - tot[0]).max() / tot[0]:.2e}, end drift = {(tot[-1] - tot[0]) / tot[0]:+.2e}")


def _dominant_omega(signal: np.ndarray, dt: float, pad: int = 16) -> float:
    """Angular frequency of the strongest spectral line of (signal - mean), via a zero-padded FFT."""
    spec = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(len(signal)), n=len(signal) * pad))
    fr = np.fft.rfftfreq(len(signal) * pad, dt) * 2 * np.pi
    return float(fr[1:][np.argmax(spec[1:])])


def gyro_orbit(args) -> None:
    """Near-zero-weight particles in a uniform B: velocities must rotate by the exact Boris angle."""
    q, m, B0, dt, nst = 1.0, 1.0, 2.0, 0.05, 400
    cfg = PICConfig(L=(32.0, 32.0, 32.0), modes=(1, 0, 0), dt=dt, c=10.0, eps0=0.01, B_eq=(0.0, 0.0, B0))
    rng = np.random.default_rng(2)
    n = 5000
    pos = rng.uniform(0, 32, (3, n))
    v0 = rng.normal(0, 1.0, (3, n))
    sp = Species(q=q, m=m, weight=1e-20)  # negligible self-fields
    sp.state = tuple(mx.array(a.astype(np.float32)) for a in (*pos, *v0))
    SpectralPIC(cfg, [sp]).run(nst, eval_every=100)
    v = np.stack([np.asarray(a, np.float64) for a in sp.state[3:]])
    ang = nst * 2 * np.arctan(0.5 * q * B0 / m * dt)  # exact Boris rotation per step, times steps
    vx = v0[0] * np.cos(ang) + v0[1] * np.sin(ang)
    vy = v0[1] * np.cos(ang) - v0[0] * np.sin(ang)
    err = max(np.abs(v[0] - vx).max(), np.abs(v[1] - vy).max(), np.abs(v[2] - v0[2]).max())
    drift = np.abs(np.linalg.norm(v, axis=0) / np.linalg.norm(v0, axis=0) - 1).max()
    print(f"[4] gyro-orbit, {nst} steps ({ang / 2 / np.pi:.2f} gyro-periods): max velocity error vs exact Boris rotation = {err:.1e}, |v| drift = {drift:.1e}")


def vacuum_wave(args) -> None:
    """No plasma: a standing light wave must oscillate at the leapfrog frequency and conserve energy."""
    L, c, dt = 64.0, 1.0, 0.05
    cfg = PICConfig(L=(L, 8.0, 8.0), modes=(3, 0, 0), dt=dt, c=c, eps0=1.0, B_eq=(0.0, 0.0, 0.0))
    A, B, C = cfg.shape
    ET0 = np.zeros((3, A, B, C), dtype=np.complex128)
    ET0[1, 1, 0, 0] = 0.5  # E_y = cos(kx)
    rng = np.random.default_rng(2)
    sp = Species(q=1.0, m=1.0, weight=1e-20)
    sp.state = tuple(mx.array(a.astype(np.float32)) for a in (*rng.uniform(0, L, (3, 100)), *np.zeros((3, 100))))
    pic = SpectralPIC(cfg, [sp], E_T0=ET0)
    k = 2 * np.pi / L
    w_num = 2 / dt * np.arcsin(c * k * dt / 2)
    steps = int(4 * 2 * np.pi / (c * k) / dt)
    pic.run(steps, eval_every=200, diag_every=1)
    h = pic.energy_history()
    w = _dominant_omega(h[:, 1], dt) / 2
    print(f"[5] vacuum light wave, {steps} steps: omega = {w:.5f}, c k = {c * k:.5f}, leapfrog theory = {w_num:.5f};"
          f" W_B reaches {h[:, 2].max() / h[0, 1]:.3f} x W_E(0); total-energy variation = {np.ptp(h.sum(axis=1)) / h.sum(axis=1).mean():.1e}")


def plasma_wave(args) -> None:
    """Transverse light wave in a cold plasma: omega^2 = omega_pe^2 + c^2 k^2 (tests the J_T -> E_T coupling)."""
    L = (64.0, 8.0, 8.0)
    c, eps0, dt, wp = 1.0, 1.0, 0.05, 0.5
    cfg = PICConfig(L=L, modes=(2, 0, 0), dt=dt, c=c, eps0=eps0, B_eq=(0.0, 0.0, 0.0))
    rng = np.random.default_rng(4)
    n = 100_000
    vol = L[0] * L[1] * L[2]
    weight = wp**2 * eps0 * vol / n  # electrons: q = -1, m = 1
    pos = rng.uniform(0, 1, (3, n)) * np.array(L)[:, None]
    ions, ele = Species(q=+1.0, m=1e9, weight=weight), Species(q=-1.0, m=1.0, weight=weight)
    for sp in (ions, ele):  # quiet start: electrons on the ions => rho = 0
        sp.state = tuple(mx.array(a.astype(np.float32)) for a in (*pos, *np.zeros((3, n))))
    A, B, C = cfg.shape
    ET0 = np.zeros((3, A, B, C), dtype=np.complex128)
    ET0[1, 1, 0, 0] = 0.05
    pic = SpectralPIC(cfg, [ions, ele], E_T0=ET0)
    k = 2 * np.pi / L[0]
    w_th = np.sqrt(wp**2 + c**2 * k**2)
    steps = int(6 * 2 * np.pi / w_th / dt)
    pic.run(steps, eval_every=100, diag_every=1)
    h = pic.energy_history()
    w = _dominant_omega(h[:, 1], dt) / 2
    print(f"[6] light wave in plasma (omega_pe={wp}), {steps} steps: omega = {w:.4f}, theory sqrt(wp^2+c^2k^2) = {w_th:.4f} "
          f"({100 * abs(w - w_th) / w_th:.2f}% off; vacuum c k would be {c * k:.4f}); total-energy variation = {np.ptp(h.sum(axis=1)) / h.sum(axis=1).mean():.1e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--modes", type=int, nargs=3, default=[3, 1, 1])
    ap.add_argument("--amp", type=float, default=2e-2)
    ap.add_argument("--thermal-steps", type=int, default=2000)
    ap.add_argument("--only", choices=["ref", "langmuir", "thermal", "gyro", "vacuum", "plasmawave"], default=None)
    args = ap.parse_args()
    if args.only in (None, "ref"):
        compare_to_reference(args)
    if args.only in (None, "langmuir"):
        langmuir(args)
    if args.only in (None, "thermal"):
        thermal_energy(args)
    if args.only in (None, "gyro"):
        gyro_orbit(args)
    if args.only in (None, "vacuum"):
        vacuum_wave(args)
    if args.only in (None, "plasmawave"):
        plasma_wave(args)


if __name__ == "__main__":
    main()
