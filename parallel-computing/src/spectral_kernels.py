"""GPU kernels for the direct 3D spectral EM/PIC step (Apple GPU, MLX custom Metal).

Design rules (all measured earlier in this project):
  * particle state is six flat float32 arrays (struct of arrays);
  * nothing of size N x K (particles x modes) is ever written to global memory;
  * exp(i k x) is evaluated ONCE per particle per axis (one sin, one cos) and the
    higher harmonics come from repeated complex multiplication, e^{i m t} = e^{i (m-1) t} e^{i t};
  * Hermitian symmetry: real fields have F_{-k} = conj(F_k), so only m_x >= 0 is
    kept (A = HX + 1 x-modes, B = 2 HY + 1, C = 2 HZ + 1).  Gather uses folded weights
    (1 for m_x = 0, 2 for m_x > 0) so that f(x) = Re sum_{m_x >= 0} F'_k e^{i k.x}.

Two kernels per species per step:
  gather_push : one thread per particle -> sums E, B over modes in registers, Boris push.
  deposit     : tiled reduction over particles -> rho_k and J_k (8 real channels).
"""

from __future__ import annotations

import mlx.core as mx

HEADER = """
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// out[m] = exp(i m theta), m = 0..M-1: ONE sin/cos, then repeated multiplication.
template <int M>
inline void harmonics(float theta, thread float2* out) {
    float2 base = float2(metal::cos(theta), metal::sin(theta));
    out[0] = float2(1.0f, 0.0f);
    #pragma unroll
    for (int m = 1; m < M; ++m) out[m] = cmul(out[m - 1], base);
}

// out[i] = exp(i (i - H) theta) for i = 0..N2-1 (m = -H..H, N2 = 2H+1); negative m by conjugation.
template <int N2>
inline void harmonics_signed(float theta, thread float2* out) {
    constexpr int H = (N2 - 1) / 2;
    float2 pos[H + 1];
    harmonics<H + 1>(theta, pos);
    #pragma unroll
    for (int i = 0; i < N2; ++i) {
        int m = i - H;
        float2 v = pos[m >= 0 ? m : -m];
        out[i] = float2(v.x, m >= 0 ? v.y : -v.y);
    }
}
"""

# ----------------------------------------------------------------------------------------
# gather + Boris push.  One thread owns P particles; a threadgroup is TPG threads.
#
#  * No per-particle phase arrays: the phase of each mode comes from RUNNING PRODUCTS,
#    p_x <- p_x * e^{i k0x x}, p_y <- p_y * e^{i k0y y}, p_z <- p_z * e^{i k0z z}, walking the
#    modes (a, b, c) in order.  Each axis needs ONE sin/cos per particle, nothing is indexed
#    dynamically, and the same code handles any mode-grid size.
#  * The field table is streamed through threadgroup memory in slabs of SA x-planes (float4
#    loads, 12 floats = E,B re/im per mode).  Every table value loaded is reused by all
#    TPG*P particles of the threadgroup.
#  * Table layout: [a][b][c][comp(E0,E1,E2,B0,B1,B2)][re,im], already folded (x2 for m_x > 0).
# params = [h = q/m*dt/2, dt, Lx, Ly, Lz, k0x, k0y, k0z, Beq_x, Beq_y, Beq_z]
# ----------------------------------------------------------------------------------------
_GATHER_SRC = """
    constexpr int BC = B * C;
    constexpr int HY = (B - 1) / 2;
    constexpr int HZ = (C - 1) / 2;

    threadgroup float4 sF4[SA * BC * 3];
    const device float4* F4 = reinterpret_cast<const device float4*>(F);

    int tid = (int)thread_position_in_threadgroup.x;
    int g = (int)threadgroup_position_in_grid.x;

    float2 xb[P], yb[P], zb[P], px[P], y0[P], z0[P];
    float3 pos[P], vel[P];
    float acc[P][6];
    bool ok[P];

    #pragma unroll
    for (int p = 0; p < P; ++p) {
        int idx = (g * P + p) * TPG + tid;
        ok[p] = idx < N;
        pos[p] = ok[p] ? float3(x[idx], y[idx], z[idx]) : float3(0.0f);
        vel[p] = ok[p] ? float3(vx[idx], vy[idx], vz[idx]) : float3(0.0f);
        // ONE sin/cos per axis per particle; everything else is multiplication
        xb[p] = float2(metal::cos(params[5] * pos[p].x), metal::sin(params[5] * pos[p].x));
        yb[p] = float2(metal::cos(params[6] * pos[p].y), metal::sin(params[6] * pos[p].y));
        zb[p] = float2(metal::cos(params[7] * pos[p].z), metal::sin(params[7] * pos[p].z));
        px[p] = float2(1.0f, 0.0f);
        float2 s = float2(1.0f, 0.0f);
        for (int i = 0; i < HY; ++i) s = cmul(s, yb[p]);
        y0[p] = float2(s.x, -s.y);                       // e^{-i HY theta_y}
        s = float2(1.0f, 0.0f);
        for (int i = 0; i < HZ; ++i) s = cmul(s, zb[p]);
        z0[p] = float2(s.x, -s.y);                       // e^{-i HZ theta_z}
        #pragma unroll
        for (int f = 0; f < 6; ++f) acc[p][f] = 0.0f;
    }

    for (int a0 = 0; a0 < A; a0 += SA) {
        int na = min(SA, A - a0);
        threadgroup_barrier(mem_flags::mem_threadgroup);          // previous slab fully consumed
        for (int i = tid; i < na * BC * 3; i += TPG) sF4[i] = F4[a0 * BC * 3 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int al = 0; al < na; ++al) {
            float2 py[P];
            #pragma unroll
            for (int p = 0; p < P; ++p) py[p] = y0[p];
            for (int b = 0; b < B; ++b) {
                float2 t[P], pz[P];
                #pragma unroll
                for (int p = 0; p < P; ++p) { t[p] = cmul(px[p], py[p]); pz[p] = z0[p]; }
                for (int c = 0; c < C; ++c) {
                    int m = ((al * B + b) * C + c) * 3;
                    float4 f0 = sF4[m], f1 = sF4[m + 1], f2 = sF4[m + 2];
                    #pragma unroll
                    for (int p = 0; p < P; ++p) {
                        float2 ph = cmul(t[p], pz[p]);
                        acc[p][0] += f0.x * ph.x - f0.y * ph.y;   // Ex
                        acc[p][1] += f0.z * ph.x - f0.w * ph.y;   // Ey
                        acc[p][2] += f1.x * ph.x - f1.y * ph.y;   // Ez
                        acc[p][3] += f1.z * ph.x - f1.w * ph.y;   // Bx
                        acc[p][4] += f2.x * ph.x - f2.y * ph.y;   // By
                        acc[p][5] += f2.z * ph.x - f2.w * ph.y;   // Bz
                        pz[p] = cmul(pz[p], zb[p]);
                    }
                }
                #pragma unroll
                for (int p = 0; p < P; ++p) py[p] = cmul(py[p], yb[p]);
            }
            #pragma unroll
            for (int p = 0; p < P; ++p) px[p] = cmul(px[p], xb[p]);
        }
    }

    float h = params[0], dt = params[1];
    #pragma unroll
    for (int p = 0; p < P; ++p) {
        float3 E = float3(acc[p][0], acc[p][1], acc[p][2]);
        float3 Bf = float3(acc[p][3] + params[8], acc[p][4] + params[9], acc[p][5] + params[10]);

        // Boris: half E kick, magnetic rotation, half E kick
        float3 vm = vel[p] + h * E;
        float3 t = h * Bf;
        float3 s = 2.0f * t / (1.0f + dot(t, t));
        float3 vp = vm + cross(vm, t);
        float3 vn = vm + cross(vp, s) + h * E;

        float3 np = pos[p] + dt * vn;
        np.x -= params[2] * metal::floor(np.x / params[2]);
        np.y -= params[3] * metal::floor(np.y / params[3]);
        np.z -= params[4] * metal::floor(np.z / params[4]);

        if (ok[p]) {
            int idx = (g * P + p) * TPG + tid;
            ox[idx] = np.x; oy[idx] = np.y; oz[idx] = np.z;
            ovx[idx] = vn.x; ovy[idx] = vn.y; ovz[idx] = vn.z;
        }
    }
"""

_gather_kernel = mx.fast.metal_kernel(
    name="spectral_gather_push",
    input_names=["x", "y", "z", "vx", "vy", "vz", "F", "params"],
    output_names=["ox", "oy", "oz", "ovx", "ovy", "ovz"],
    source=_GATHER_SRC,
    header=HEADER,
)

# ----------------------------------------------------------------------------------------
# deposit:  NQ = 1 -> rho_k = sum q w e^{-ik.x};   NQ = 3 -> J_k = sum q w v e^{-ik.x} (3 components).
# Threadgroup (g, ac) owns particle chunk g and the x-modes [ac*AC, ac*AC+AC).  Thread (s, bc)
# owns one (b,c) mode pair and a particle sub-lane s, and keeps AC x-modes x NQ quantities x
# (re,im) = AC*2*NQ accumulators in registers -- a FIXED number, whatever the mode-grid size
# (without the x-chunking the accumulators grow with A and spill: 14x slower per mode at A=17).
# params = [q*w, k0x, k0y, k0z, dtm]   (deposit at x - dtm*v: dtm = 0 -> x^n, dtm = dt/2 -> mid-step)
# out layout: [group][channel(2*NQ)][a][b][c]; channel = quantity*2 + (re, im)
# ----------------------------------------------------------------------------------------
_DEPOSIT_SRC = """
    constexpr int BC = B * C;
    constexpr int K = A * BC;
    constexpr int T = S * BC;
    constexpr int NC = 2 * NQ;
    constexpr bool VEL = (NQ == 3);
    constexpr int OFF = ((TILE * (3 + NQ) + 1) / 2) * 2;

    uint g = threadgroup_position_in_grid.x;
    int a0 = (int)threadgroup_position_in_grid.y * AC;      // first x-mode of this chunk
    int tid = (int)thread_position_in_threadgroup.x;

    threadgroup float4 smem4[(SMEM + 3) / 4];
    threadgroup float* smem = reinterpret_cast<threadgroup float*>(smem4);
    threadgroup float* shcx = smem;
    threadgroup float* shcy = shcx + TILE;
    threadgroup float* shcz = shcy + TILE;
    threadgroup float* shW = shcz + TILE;                                   // NQ * TILE
    threadgroup float2* shPx = reinterpret_cast<threadgroup float2*>(smem + OFF);
    threadgroup float2* shPy = shPx + TILE * AC;
    threadgroup float2* shPz = shPy + TILE * B;
    threadgroup float* red = smem;                                          // reused after the last tile

    float qw = params[0];
    float k0x = params[1], k0y = params[2], k0z = params[3];
    float dtm = params[4];     // deposit at x - dtm * v (dtm = dt/2 -> mid-step position; unwrapped is fine, e^{ikx} is periodic)

    int start = (int)g * CH;
    int end = min((int)N, start + CH);

    int s = tid / BC;
    int bc = tid - s * BC;
    int b = bc / C;
    int c = bc - b * C;

    float acc[AC * NC];
    #pragma unroll
    for (int i = 0; i < AC * NC; ++i) acc[i] = 0.0f;

    // software prefetch: one coalesced load per particle, one tile ahead
    float px = 0.0f, py = 0.0f, pz = 0.0f, pvx = 0.0f, pvy = 0.0f, pvz = 0.0f;
    if (tid < TILE) {
        int p = start + tid;
        if (p < end) { px = x[p]; py = y[p]; pz = z[p]; pvx = vx[p]; pvy = vy[p]; pvz = vz[p]; }
    }

    for (int t0 = start; t0 < end; t0 += TILE) {
        if (tid < TILE) {
            float wgt = (t0 + tid < end) ? qw : 0.0f;                       // padded lanes weigh 0
            shcx[tid] = px - dtm * pvx; shcy[tid] = py - dtm * pvy; shcz[tid] = pz - dtm * pvz;
            if (VEL) {
                shW[NQ * tid] = wgt * pvx;
                shW[NQ * tid + 1] = wgt * pvy;
                shW[NQ * tid + 2] = wgt * pvz;
            } else {
                shW[tid] = wgt;
            }
            px = 0.0f; py = 0.0f; pz = 0.0f; pvx = 0.0f; pvy = 0.0f; pvz = 0.0f;
            int pn = t0 + TILE + tid;
            if (pn < end) { px = x[pn]; py = y[pn]; pz = z[pn]; pvx = vx[pn]; pvy = vy[pn]; pvz = vz[pn]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- map: per (particle, axis) one sin/cos + recurrence -> on-chip phase tables ----
        for (int idx = tid; idx < TILE * 3; idx += T) {
            int j = idx / 3;
            int ax = idx - j * 3;
            if (ax == 0) {
                // ONE sin/cos; e^{i a0 t} by binary exponentiation (O(log a0) multiplications; a linear
                // prefix cost up to a0 = 128 multiplications per particle per chunk), the chunk by AC more
                float th = k0x * shcx[j];
                float2 base = float2(metal::cos(th), metal::sin(th));
                float2 cur = float2(1.0f, 0.0f);
                { float2 pw = base; int e = a0; while (e > 0) { if (e & 1) cur = cmul(cur, pw); pw = cmul(pw, pw); e >>= 1; } }
                #pragma unroll
                for (int m = 0; m < AC; ++m) { shPx[j * AC + m] = cur; cur = cmul(cur, base); }
            } else if (ax == 1) {
                float2 ph[B];
                harmonics_signed<B>(k0y * shcy[j], ph);
                for (int m = 0; m < B; ++m) shPy[j * B + m] = ph[m];
            } else {
                float2 ph[C];
                harmonics_signed<C>(k0z * shcz[j], ph);
                for (int m = 0; m < C; ++m) shPz[j * C + m] = ph[m];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- accumulate in registers: sum_j W_q conj(Px[a] Py[b] Pz[c]) ----
        for (int j = s; j < TILE; j += S) {
            float2 yz = cmul(shPy[j * B + b], shPz[j * C + c]);
            float wq[NQ];
            #pragma unroll
            for (int qi = 0; qi < NQ; ++qi) wq[qi] = shW[NQ * j + qi];
            #pragma unroll
            for (int a = 0; a < AC; ++a) {
                float2 t = cmul(shPx[j * AC + a], yz);
                #pragma unroll
                for (int qi = 0; qi < NQ; ++qi) {
                    acc[a * NC + 2 * qi] += wq[qi] * t.x;
                    acc[a * NC + 2 * qi + 1] -= wq[qi] * t.y;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // shW / phase tables are rewritten next tile
    }

    // ---- sum the S sub-lanes on-chip; one partial result per threadgroup goes to global memory ----
    #pragma unroll
    for (int a = 0; a < AC; ++a)
        #pragma unroll
        for (int ch = 0; ch < NC; ++ch)
            red[((s * BC + bc) * AC + a) * NC + ch] = acc[a * NC + ch];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = tid; idx < BC * AC * NC; idx += T) {
        int ch = idx % NC;
        int t2 = idx / NC;
        int al = t2 % AC;
        int q = t2 / AC;
        if (a0 + al < A) {
            float sum = 0.0f;
            for (int s2 = 0; s2 < S; ++s2) sum += red[s2 * (BC * AC * NC) + idx];
            out[(int)g * NC * K + ch * K + (a0 + al) * BC + q] = sum;
        }
    }
"""

_deposit_kernel = mx.fast.metal_kernel(
    name="spectral_deposit",
    input_names=["x", "y", "z", "vx", "vy", "vz", "params"],
    output_names=["out"],
    source=_DEPOSIT_SRC,
    header=HEADER,
)

_SHARED_LIMIT_BYTES = 30_000  # Apple GPUs: 32 KB threadgroup memory


def gather_push(state, table, params, shape, tpg: int = 128, ppt: int = 2):
    """state = (x, y, z, vx, vy, vz); table in the [a][b][c][comp][re,im] layout; returns the pushed state.

    tpg = threads per threadgroup, ppt = particles per thread.  The field table is streamed
    through threadgroup memory in slabs of x-planes, so any grid whose single x-plane
    (B*C modes * 48 bytes) fits in ~24 KB works.
    """
    A, B, C = shape
    plane_bytes = B * C * 48
    if plane_bytes > 24_000:
        raise ValueError(f"one x-plane of the mode table ({plane_bytes} B) exceeds threadgroup memory; reduce hy/hz")
    sa = max(1, min(A, 24_000 // plane_bytes))
    n = state[0].shape[0]
    groups = -(-n // (tpg * ppt))
    return tuple(
        _gather_kernel(
            inputs=[*state, table, params],
            template=[("A", A), ("B", B), ("C", C), ("N", n), ("TPG", tpg), ("P", ppt), ("SA", sa)],
            grid=(groups * tpg, 1, 1),
            threadgroup=(tpg, 1, 1),
            output_shapes=[(n,)] * 6,
            output_dtypes=[mx.float32] * 6,
        )
    )


def _smem_floats(ac: int, b: int, c: int, s: int, tile: int, nq: int) -> int:
    off = ((tile * (3 + nq) + 1) // 2) * 2
    return max(off + 2 * tile * (ac + b + c), s * b * c * ac * 2 * nq)


class DepositPlan:
    """Threadgroup geometry for the tiled deposit reduction.

    nq = quantities deposited (1: rho, 3: J); ac = x-modes per threadgroup pass
    (register accumulators = ac * 2 * nq per thread).  Default ac, from sweeps over three mode geometries:
    8 for rho (1.2-1.6x faster than 4) and 4 for J (AC >= 12 spills registers: 7x slower).
    Limits: (2hy+1)(2hz+1) <= 1024 threads per threadgroup.
    """

    def __init__(self, shape, n: int, nq: int = 3, groups: int | None = None, ac: int | None = None, tile_mult: int = 8) -> None:
        self.shape = tuple(shape)
        self.n = n
        self.NQ = nq
        A, B, C = self.shape
        bc = B * C
        if bc > 1024:
            raise ValueError(f"(2hy+1)(2hz+1) = {bc} exceeds 1024 threads per threadgroup")
        self.AC = min(ac if ac is not None else (8 if nq == 1 else 4), A)
        self.nch = -(-A // self.AC)
        # Measured optimum: ~2000 particles per threadgroup, but keep enough groups to fill the GPU.
        self.groups = groups if groups is not None else max(32, min(2048, n // 2000))
        self.chunk = -(-n // self.groups)
        for s in range(max(1, 256 // bc), 0, -1):
            for tm in range(min(tile_mult, bc), 0, -1):
                tile = s * tm
                smem = _smem_floats(self.AC, B, C, s, tile, nq)
                if 4 * smem <= _SHARED_LIMIT_BYTES:
                    self.S, self.TILE, self.SMEM = s, tile, smem
                    self.T = s * bc
                    return
        raise ValueError("mode grid too large for threadgroup memory")

    def template(self):
        A, B, C = self.shape
        return [("A", A), ("B", B), ("C", C), ("AC", self.AC), ("NQ", self.NQ), ("N", self.n), ("CH", self.chunk),
                ("S", self.S), ("TILE", self.TILE), ("SMEM", self.SMEM)]


def deposit(state, params, plan: DepositPlan):
    """Return the deposited moments as a (2*NQ, A, B, C) array, channel = quantity*2 + (re, im):
    NQ=1 -> rho_k;  NQ=3 -> (Jx, Jy, Jz).  Summed over threadgroups (tiny second stage)."""
    A, B, C = plan.shape
    nc = 2 * plan.NQ
    (out,) = _deposit_kernel(
        inputs=[*state, params],
        template=plan.template(),
        grid=(plan.groups * plan.T, plan.nch, 1),
        threadgroup=(plan.T, 1, 1),
        output_shapes=[(plan.groups * nc * A * B * C,)],
        output_dtypes=[mx.float32],
    )
    return mx.sum(out.reshape(plan.groups, nc, A, B, C), axis=0)
