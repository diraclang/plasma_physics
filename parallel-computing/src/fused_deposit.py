"""Fused mode deposition: rho[a,b,c] = sum_p w_p * exp(-i (kx_a x_p + ky_b y_p + kz_c z_p)).

Nothing per-particle-per-mode is ever written to global memory.

Work decomposition
------------------
* The particle list is cut into G chunks; one threadgroup owns one chunk.
* Inside a threadgroup, thread (s, bc) owns the (b, c) mode pair `bc` and the
  particle sub-lane `s`.  It keeps the A partial sums for all x-modes of its
  (b, c) pair in registers (acc_re[A], acc_im[A]).
* Per tile of TILE particles the threadgroup cooperatively evaluates the
  *separable* phases (cos/sin of kx*x, ky*y, kz*z, i.e. A+B+C trig pairs per
  particle instead of A*B*C) into threadgroup (on-chip) memory, with the
  particle weight folded into the x factor.
* Each thread then multiplies the y- and z-phase once and sweeps the A x-modes:
  ~13 flops per particle-mode, all operands in registers / on-chip memory.
* At the end the S sub-lanes of a threadgroup are summed on-chip and only
  2*K floats per threadgroup are written out; a tiny second stage sums the G
  partial results.

Global-memory traffic is therefore just 16 B/particle in (x, y, z, w) plus
G*2*K*4 B out.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    constexpr int BC = B * C;
    constexpr int K = A * BC;
    constexpr int NM = A + B + C;
    constexpr int U = BC / Q;      // threads that together cover all (b,c) pairs
    constexpr int T = S * U;       // threads per threadgroup

    uint g = threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;

    threadgroup float smem[SMEM];
    threadgroup float* shcx = smem;
    threadgroup float* shcy = shcx + TILE;
    threadgroup float* shcz = shcy + TILE;
    threadgroup float* shcw = shcz + TILE;
    threadgroup float* shxc = shcw + TILE;
    threadgroup float* shxs = shxc + TILE * A;
    threadgroup float* shyc = shxs + TILE * A;
    threadgroup float* shys = shyc + TILE * B;
    threadgroup float* shzc = shys + TILE * B;
    threadgroup float* shzs = shzc + TILE * C;
    threadgroup float* red = smem;   // reused after the last tile

    int start = (int)g * CH;
    int end = min((int)N, start + CH);

    int s = tid / U;
    int u = tid - s * U;

    // each thread owns Q consecutive (b,c) pairs x all A x-modes, in registers
    float acc_re[Q * A];
    float acc_im[Q * A];
    for (int i = 0; i < Q * A; ++i) { acc_re[i] = 0.0f; acc_im[i] = 0.0f; }

    // software prefetch: one coalesced load per particle, one tile ahead
    float px = 0.0f, py = 0.0f, pz = 0.0f, pw = 0.0f;
    if (tid < TILE) {
        int p = start + tid;
        if (p < end) { px = x[p]; py = y[p]; pz = z[p]; pw = w[p]; }
    }

    for (int t0 = start; t0 < end; t0 += TILE) {
        if (tid < TILE) {
            shcx[tid] = px; shcy[tid] = py; shcz[tid] = pz; shcw[tid] = pw;
            px = 0.0f; py = 0.0f; pz = 0.0f; pw = 0.0f;
            int pn = t0 + TILE + tid;
            if (pn < end) { px = x[pn]; py = y[pn]; pz = z[pn]; pw = w[pn]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- map: separable phases from on-chip coords -> on-chip phases ----
        for (int idx = tid; idx < TILE * NM; idx += T) {
            int j = idx / NM;
            int r = idx - j * NM;
            if (r < A) {
                float ang = kx[r] * shcx[j];
                float ww = shcw[j];
                shxc[j * A + r] = ww * metal::cos(ang);
                shxs[j * A + r] = ww * metal::sin(ang);
            } else if (r < A + B) {
                int m = r - A;
                float ang = ky[m] * shcy[j];
                shyc[j * B + m] = metal::cos(ang);
                shys[j * B + m] = metal::sin(ang);
            } else {
                int m = r - A - B;
                float ang = kz[m] * shcz[j];
                shzc[j * C + m] = metal::cos(ang);
                shzs[j * C + m] = metal::sin(ang);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- accumulate: everything below lives in registers ----
        for (int j = s; j < TILE; j += S) {
            float yzr[Q];
            float yzi[Q];
            for (int q = 0; q < Q; ++q) {
                int bcq = u * Q + q;
                int b = bcq / C;
                int c = bcq - b * C;
                float cy = shyc[j * B + b];
                float sy = shys[j * B + b];
                float cz = shzc[j * C + c];
                float sz = shzs[j * C + c];
                yzr[q] = cy * cz - sy * sz;
                yzi[q] = -(cy * sz + sy * cz);
            }
            for (int a = 0; a < A; ++a) {
                float cx = shxc[j * A + a];
                float sx = shxs[j * A + a];
                for (int q = 0; q < Q; ++q) {
                    acc_re[q * A + a] += cx * yzr[q] + sx * yzi[q];
                    acc_im[q * A + a] += cx * yzi[q] - sx * yzr[q];
                }
            }
        }
        // no trailing barrier: the next iteration's first barrier orders this
        // accumulate before the next map writes to the phase arrays.
    }

    // ---- reduce the S sub-lanes on-chip, write one partial per threadgroup ----
    threadgroup_barrier(mem_flags::mem_threadgroup);   // red aliases the tile buffers
    for (int q = 0; q < Q; ++q) {
        int bcq = u * Q + q;
        for (int a = 0; a < A; ++a) {
            int base = ((s * BC + bcq) * A + a) * 2;
            red[base] = acc_re[q * A + a];
            red[base + 1] = acc_im[q * A + a];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = tid; idx < BC * A * 2; idx += T) {
        int comp = idx & 1;
        int t2 = idx >> 1;
        int a = t2 % A;
        int q = t2 / A;
        float sum = 0.0f;
        for (int s2 = 0; s2 < S; ++s2) sum += red[s2 * (BC * A * 2) + idx];
        out[(int)g * 2 * K + comp * K + a * BC + q] = sum;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="fused_mode_deposit",
    input_names=["x", "y", "z", "w", "kx", "ky", "kz"],
    output_names=["out"],
    source=_SOURCE,
)

_SHARED_LIMIT_BYTES = 30_000  # Apple GPUs: 32 KB of threadgroup memory


def smem_floats(shape: tuple[int, int, int], s: int, tile: int) -> int:
    """Peak threadgroup floats: tile buffers during the loop, reduce buffer at the end (aliased)."""
    A, B, C = shape
    return max(tile * (2 * (A + B + C) + 4), s * B * C * A * 2)


def plan(shape: tuple[int, int, int], q: int = 1, tile_mult: int = 8) -> tuple[int, int]:
    """Pick (S, TILE): threadgroup memory must fit, TILE <= threads per group.

    q = number of (b,c) mode pairs each thread keeps in registers (must divide B*C).
    """
    A, B, C = shape
    BC = B * C
    if BC % q:
        raise ValueError(f"q={q} must divide B*C={BC}")
    U = BC // q
    for S in range(max(1, 256 // U), 0, -1):
        for tm in range(min(tile_mult, U), 0, -1):
            tile = S * tm
            if 4 * smem_floats(shape, S, tile) <= _SHARED_LIMIT_BYTES:
                return S, tile
    raise ValueError("mode grid too large for threadgroup memory")


class FusedDeposit:
    def __init__(self, ks: list, n: int, groups: int | None = None, q: int = 1) -> None:
        self.shape = tuple(len(k) for k in ks)
        self.n = n
        # Measured optimum: ~2000 particles per threadgroup (N=100k -> 64 ... N=4M -> 2048 groups).
        self.groups = groups if groups is not None else max(32, min(2048, n // 2000))
        A, B, C = self.shape
        self.Q = q
        self.S, self.TILE = plan(self.shape, q)
        self.T = self.S * (B * C // q)
        self.chunk = -(-n // self.groups)
        self.kx, self.ky, self.kz = (mx.array(k, dtype=mx.float32) for k in ks)

    def __call__(self, x, y, z, w):
        A, B, C = self.shape
        K = A * B * C
        (out,) = _kernel(
            inputs=[x, y, z, w, self.kx, self.ky, self.kz],
            template=[
                ("A", A), ("B", B), ("C", C),
                ("N", self.n), ("CH", self.chunk),
                ("S", self.S), ("TILE", self.TILE), ("Q", self.Q),
                ("SMEM", smem_floats(self.shape, self.S, self.TILE)),
            ],
            grid=(self.groups * self.T, 1, 1),
            threadgroup=(self.T, 1, 1),
            output_shapes=[(self.groups * 2 * K,)],
            output_dtypes=[mx.float32],
        )
        rho = mx.sum(out.reshape(self.groups, 2, A, B, C), axis=0)  # tiny 2nd stage
        return rho[0], rho[1]
