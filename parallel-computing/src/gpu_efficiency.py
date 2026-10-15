"""GPU efficiency diagnostics for the particle push (map) and the mode deposition (reduce).

Question answered: are we limited by (a) the CPU "master" (launch/dispatch overhead),
(b) memory traffic, or (c) arithmetic - and how close are we to the hardware ceiling?

Method (roofline style):
  1. Measure this machine's ceilings empirically: memory bandwidth and GEMM TFLOP/s.
  2. Push kernel (the "map"): compare achieved GB/s with the bandwidth ceiling, and
     split runtime into a fixed per-step floor (CPU dispatch) and N-proportional work.
  3. Deposition onto Fourier modes (the "reduce"): two formulations, validated against
     a float64 NumPy direct sum, timed, and compared with the same ceilings.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mlx.core as mx

from .fused_deposit import FusedDeposit
from .magnetic_gpu_solver import Config, time_gpu

mx.set_default_device(mx.gpu)


def best_time(fn, reps: int = 5) -> float:
    fn()  # warm-up / compile
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ----------------------------------------------------------------------------
# 1. Machine ceilings
# ----------------------------------------------------------------------------
def measure_ceilings() -> dict:
    n = 32 * 1024 * 1024  # 128 MB float32
    a = mx.random.normal((n,))
    mx.eval(a)
    axpy = mx.compile(lambda v: v * 1.0001 + 1e-6)
    reps = 20

    def run_axpy():
        # Dependent chain: MLX is lazy, so independent calls on the same input
        # would be deduplicated/skipped and report an impossible bandwidth.
        o = a
        for _ in range(reps):
            o = axpy(o)
        mx.eval(o)

    t_axpy = best_time(run_axpy) / reps
    bw_copy = 2 * n * 4 / t_axpy / 1e9  # read + write

    m = 4096
    A = mx.random.normal((m, m))
    B = mx.random.normal((m, m))
    mx.eval(A, B)

    def run_gemm():
        mx.eval(A @ B)

    t_gemm = best_time(run_gemm, reps=5)
    tflops = 2 * m**3 / t_gemm / 1e12

    return {"bw_peak": bw_copy, "gemm_tflops": tflops}


# ----------------------------------------------------------------------------
# 2. Push kernel
# ----------------------------------------------------------------------------
def push_report(ceil: dict, steps: int, sizes: list[int]) -> dict:
    cfg = Config()
    bytes_pp = 48  # 6 float32 read + 6 float32 written (conservative)
    flops_pp = 20
    floor = time_gpu(1000, steps, cfg)  # tiny N: pure dispatch/launch cost
    print(f"\n[push] fixed per-step floor (N=1000, CPU dispatch + launch): {floor * 1e3:.4f} ms")
    print(f"{'N':>10} {'ms/step':>9} {'GB/s':>8} {'%BW peak':>9} {'GFLOP/s':>9} {'%fixed overhead':>16}")
    out = {}
    for n in sizes:
        t = time_gpu(n, steps, cfg)
        gbs = n * bytes_pp / t / 1e9
        gfl = n * flops_pp / t / 1e9
        over = min(1.0, floor / t)
        out[n] = t
        print(f"{n:>10} {t * 1e3:>9.3f} {gbs:>8.1f} {100 * gbs / ceil['bw_peak']:>8.0f}% {gfl:>9.1f} {100 * over:>15.0f}%")
    print(
        f"[push] arithmetic intensity = {flops_pp / bytes_pp:.2f} FLOP/byte  "
        f"(GPU balance point ~ {ceil['gemm_tflops'] * 1e3 / ceil['bw_peak']:.0f} FLOP/byte) -> memory-bound by design"
    )
    return out


# ----------------------------------------------------------------------------
# 3. Mode deposition (the reduction)
# ----------------------------------------------------------------------------
def make_modes(shape: tuple[int, int, int]):
    ks = [np.arange(-(s // 2), s // 2 + 1, dtype=np.float32) * (2 * np.pi) for s in shape]
    return ks  # box size 1 -> k = 2*pi*m


def deposit_numpy_f64(pos, w, ks):
    """Direct float64 reference: rho[a,b,c] = sum_p w exp(-i k.x)."""
    p = pos.astype(np.float64)
    ex = np.exp(-1j * np.outer(p[:, 0], ks[0]))
    ey = np.exp(-1j * np.outer(p[:, 1], ks[1]))
    ez = np.exp(-1j * np.outer(p[:, 2], ks[2]))
    return np.einsum("p,pa,pb,pc->abc", w.astype(np.float64), ex, ey, ez, optimize=True)


def build_gemm_deposit(ks):
    kx, ky, kz = (mx.array(k) for k in ks)
    A, B, C = len(ks[0]), len(ks[1]), len(ks[2])

    def deposit(x, y, z, w):
        ax, ay, az = x[:, None] * kx, y[:, None] * ky, z[:, None] * kz
        cx, sx = mx.cos(ax) * w[:, None], mx.sin(ax) * w[:, None]
        cy, sy, cz, sz = mx.cos(ay), mx.sin(ay), mx.cos(az), mx.sin(az)
        # (cy - i sy)(cz - i sz) = R + i I
        R = (cy[:, :, None] * cz[:, None, :] - sy[:, :, None] * sz[:, None, :]).reshape(-1, B * C)
        I = (-(cy[:, :, None] * sz[:, None, :] + sy[:, :, None] * cz[:, None, :])).reshape(-1, B * C)
        X = mx.concatenate([cx, sx], axis=1)  # (N, 2A)
        Y = mx.concatenate([R, I], axis=1)  # (N, 2BC)
        G = X.T @ Y  # (2A, 2BC): the reduction over N is a GEMM
        BC = B * C
        rho_r = G[:A, :BC] + G[A:, BC:]
        rho_i = G[:A, BC:] - G[A:, :BC]
        return rho_r.reshape(A, B, C), rho_i.reshape(A, B, C)

    return mx.compile(deposit)


def build_naive_deposit(ks):
    kx, ky, kz = np.meshgrid(*ks, indexing="ij")
    kx, ky, kz = (mx.array(k.reshape(-1)) for k in (kx, ky, kz))
    shape = tuple(len(k) for k in ks)

    def deposit(x, y, z, w):
        ang = x[:, None] * kx + y[:, None] * ky + z[:, None] * kz  # (N, K)
        rho_r = mx.sum(w[:, None] * mx.cos(ang), axis=0)
        rho_i = -mx.sum(w[:, None] * mx.sin(ang), axis=0)
        return rho_r.reshape(shape), rho_i.reshape(shape)

    return mx.compile(deposit)


def deposit_report(ceil: dict, push_t: dict, shape, sizes, naive_max: int) -> None:
    ks = make_modes(shape)
    A, B, C = (len(k) for k in ks)
    K = A * B * C
    print(f"\n[deposit] mode grid {A}x{B}x{C} = {K} modes")

    rng = np.random.default_rng(3)
    # correctness at moderate N against float64 direct sum
    n_chk = 50_000
    pos = rng.uniform(0, 1, (n_chk, 3)).astype(np.float32)
    w = rng.normal(1.0, 0.1, n_chk).astype(np.float32)
    ref = deposit_numpy_f64(pos, w, ks)
    scale = np.max(np.abs(ref))
    cols = [mx.array(pos[:, i]) for i in range(3)] + [mx.array(w)]
    for name, builder in (("gemm", build_gemm_deposit), ("naive", build_naive_deposit), ("fused", None)):
        fn = FusedDeposit(ks, n_chk) if name == "fused" else builder(ks)
        rr, ri = fn(*cols)
        got = np.array(rr) + 1j * np.array(ri)
        print(f"  check {name:5s}: max |err| / max|rho| = {np.max(np.abs(got - ref)) / scale:.2e}")

    print(f"{'N':>10} {'form':>6} {'ms':>9} {'GFLOP/s':>9} {'%GEMM peak':>11} {'GB/s':>8} {'%BW peak':>9} {'x push':>8}")
    for n in sizes:
        pos = rng.uniform(0, 1, (n, 3)).astype(np.float32)
        w = rng.normal(1.0, 0.1, n).astype(np.float32)
        cols = [mx.array(pos[:, i]) for i in range(3)] + [mx.array(w)]
        mx.eval(*cols)
        for name, builder in (("gemm", build_gemm_deposit), ("naive", build_naive_deposit), ("fused", None)):
            if name == "naive" and n > naive_max:
                continue
            fn = FusedDeposit(ks, n) if name == "fused" else builder(ks)

            def run():
                mx.eval(*fn(*cols))

            t = best_time(run, reps=3)
            if name == "gemm":
                flops = 8.0 * n * K  # GEMM (2A)x(2BC) over N
                bytes_ = 2.0 * n * (2 * B * C + 2 * A) * 4  # materialised X,Y written then read
            elif name == "fused":
                flops = 13.0 * n * K  # complex phase product + accumulate, in registers
                bytes_ = n * 16.0  # only x,y,z,w are read from global memory
            else:
                flops = 8.0 * n * K
                bytes_ = 6.0 * n * K * 4  # several (N,K) temporaries
            ratio = t / push_t.get(n, np.nan)
            print(
                f"{n:>10} {name:>6} {t * 1e3:>9.2f} {flops / t / 1e9:>9.0f} "
                f"{100 * flops / t / 1e12 / ceil['gemm_tflops']:>10.1f}% {bytes_ / t / 1e9:>8.1f} "
                f"{100 * bytes_ / t / 1e9 / ceil['bw_peak']:>8.0f}% {ratio:>8.1f}"
            )

    split_map_reduce(ks, ceil, sizes[-1])


def split_map_reduce(ks, ceil: dict, n: int) -> None:
    """Time the deposition's map half (trig + products -> materialised X, Y) and
    reduce half (X^T Y over the particle axis) separately."""
    A, B, C = (len(k) for k in ks)
    BC = B * C
    kx, ky, kz = (mx.array(k) for k in ks)

    def phases(x, y, z, w):
        ax, ay, az = x[:, None] * kx, y[:, None] * ky, z[:, None] * kz
        cx, sx = mx.cos(ax) * w[:, None], mx.sin(ax) * w[:, None]
        cy, sy, cz, sz = mx.cos(ay), mx.sin(ay), mx.cos(az), mx.sin(az)
        R = (cy[:, :, None] * cz[:, None, :] - sy[:, :, None] * sz[:, None, :]).reshape(-1, BC)
        I = (-(cy[:, :, None] * sz[:, None, :] + sy[:, :, None] * cz[:, None, :])).reshape(-1, BC)
        return mx.concatenate([cx, sx], axis=1), mx.concatenate([R, I], axis=1)

    cph = mx.compile(phases)
    rng = np.random.default_rng(0)
    cols = [mx.array(rng.uniform(0, 1, n).astype(np.float32)) for _ in range(3)]
    cols.append(mx.array(rng.normal(1.0, 0.1, n).astype(np.float32)))
    mx.eval(*cols)
    X, Y = cph(*cols)
    mx.eval(X, Y)

    t_map = best_time(lambda: mx.eval(*cph(*cols)))
    map_bytes = n * (4 * 4 + (2 * A + 2 * BC) * 4)  # read 4 floats, write X and Y
    print(f"\n[deposit split, N={n}]")
    print(f"  MAP    (trig + products, writes X,Y): {t_map * 1e3:8.2f} ms  {map_bytes / t_map / 1e9:6.1f} GB/s  {100 * map_bytes / t_map / 1e9 / ceil['bw_peak']:4.0f}% of BW peak")
    red_bytes = n * (2 * A + 2 * BC) * 4  # re-read X and Y
    for g in (1, 256):
        if n % g:
            continue
        Xg = X.reshape(g, n // g, 2 * A)
        Yg = Y.reshape(g, n // g, 2 * BC)

        def red():
            mx.eval(mx.sum(mx.matmul(mx.swapaxes(Xg, 1, 2), Yg), axis=0))

        t = best_time(red)
        print(f"  REDUCE (X^T Y over N, {g:>3} chunk{'s' if g > 1 else ' '}):        {t * 1e3:8.2f} ms  {red_bytes / t / 1e9:6.1f} GB/s  {100 * red_bytes / t / 1e9 / ceil['bw_peak']:4.0f}% of BW peak  {8.0 * n * A * BC / t / 1e12:5.2f} TFLOP/s")
    min_bytes = n * 16
    print(f"  bandwidth floor if X,Y were never materialised (read 16 B/particle only): {min_bytes / (ceil['bw_peak'] * 1e9) * 1e3:.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU efficiency diagnostics")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--modes", type=int, nargs=3, default=[7, 7, 3], help="retained modes per axis")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 4_000_000])
    ap.add_argument("--naive-max", type=int, default=1_000_000)
    args = ap.parse_args()

    ceil = measure_ceilings()
    print("[ceilings] memory copy (R+W): %.0f GB/s | GEMM fp32: %.2f TFLOP/s" % (ceil["bw_peak"], ceil["gemm_tflops"]))
    push_t = push_report(ceil, args.steps, args.sizes)
    deposit_report(ceil, push_t, tuple(args.modes), args.sizes, args.naive_max)


if __name__ == "__main__":
    main()
