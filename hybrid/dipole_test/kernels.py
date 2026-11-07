"""Optional Numba kernels for the particle <-> grid work of the hybrid solver.

Two parallel kernels replace the NumPy stencil/deposit/gather/push path in
`simulate.run_particle_ensemble_self_consistent`:

* `deposit`: cloud-in-cell deposit of (count, ur, uphi, uz) sums.  Each thread
  accumulates into a private grid, then the grids are reduced (no write conflicts).
* `push`: per particle, gather E and dB from the grid, add the analytic background
  field (dipole + uniform + outer + toroidal) and optional corotation E, then take a
  Boris step.

They implement exactly the same arithmetic as the NumPy path (same stencil, weights and
background-field formulas); only the summation order of the deposit differs.
Numba is optional: `HAVE_NUMBA` is False when it cannot be imported.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised implicitly depending on the environment
    import numba as nb
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    nb = None
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _deposit_kernel(pos, vel, r0, z0, dr, dz, nr, nphi, nz, buf):
        nchunk = buf.shape[0]
        ncell = nr * nphi * nz
        n = pos.shape[0]
        chunk = (n + nchunk - 1) // nchunk
        dphi = 2.0 * math.pi / nphi
        two_pi = 2.0 * math.pi
        for c in prange(nchunk):
            b = buf[c]
            b[:, :] = 0.0
            lo = c * chunk
            hi = min(n, lo + chunk)
            for p in range(lo, hi):
                x = pos[p, 0]
                y = pos[p, 1]
                z = pos[p, 2]
                r = math.sqrt(x * x + y * y)
                fr = (r - r0) / dr
                fz = (z - z0) / dz
                ir = int(math.floor(fr))
                iz = int(math.floor(fz))
                if ir < 0 or ir >= nr - 1 or iz < 0 or iz >= nz - 1:
                    continue
                if r > 0.0:
                    cp = x / r
                    sp = y / r
                else:
                    cp = 1.0
                    sp = 0.0
                ur = vel[p, 0] * cp + vel[p, 1] * sp
                uphi = -vel[p, 0] * sp + vel[p, 1] * cp
                uz = vel[p, 2]

                wr1 = fr - ir
                wz1 = fz - iz
                wr0 = 1.0 - wr1
                wz0 = 1.0 - wz1

                if nphi == 1:
                    nplane = 1
                    ip0 = 0
                    ip1 = 0
                    wp0 = 1.0
                    wp1 = 0.0
                else:
                    nplane = 2
                    phi = math.atan2(y, x) % two_pi
                    fp = phi / dphi
                    ip0 = int(math.floor(fp))
                    wp1 = fp - ip0
                    ip0 = ip0 % nphi
                    ip1 = (ip0 + 1) % nphi
                    wp0 = 1.0 - wp1

                for q in range(nplane):
                    ip = ip0 if q == 0 else ip1
                    wp = wp0 if q == 0 else wp1
                    for corner in range(4):
                        ii = ir + (corner & 1)
                        kk = iz + (corner >> 1)
                        wr = wr1 if (corner & 1) else wr0
                        wz = wz1 if (corner >> 1) else wz0
                        w = wr * wz
                        if nphi > 1:
                            w = w * wp
                        cell = (ii * nphi + ip) * nz + kk
                        b[0, cell] += w
                        b[1, cell] += w * ur
                        b[2, cell] += w * uphi
                        b[3, cell] += w * uz

        out = np.empty((4, ncell))
        for cell in prange(ncell):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for c in range(nchunk):
                s0 += buf[c, 0, cell]
                s1 += buf[c, 1, cell]
                s2 += buf[c, 2, cell]
                s3 += buf[c, 3, cell]
            out[0, cell] = s0
            out[1, cell] = s1
            out[2, cell] = s2
            out[3, cell] = s3
        return out

    @njit(parallel=True, cache=True)
    def _push_kernel(
        pos, vel, fields, r0, z0, dr, dz, nr, nphi, nz,
        m_vec, ubz, outer_bz, outer_r0, outer_w, tor_b, tor_r0,
        with_coro, omega, q_over_m, dt,
        e_out, b_out, x_new, v_new,
    ):
        n = pos.shape[0]
        dphi = 2.0 * math.pi / nphi
        two_pi = 2.0 * math.pi
        width = max(outer_w, 1e-6)
        r_core = max(tor_r0, 1e-6)
        for p in prange(n):
            x = pos[p, 0]
            y = pos[p, 1]
            z = pos[p, 2]
            r = math.sqrt(x * x + y * y)
            if r > 0.0:
                cp = x / r
                sp = y / r
            else:
                cp = 1.0
                sp = 0.0

            # ---- gather the six grid components (E_r, E_phi, E_z, dB_r, dB_phi, dB_z)
            g0 = 0.0
            g1 = 0.0
            g2 = 0.0
            g3 = 0.0
            g4 = 0.0
            g5 = 0.0
            fr = (r - r0) / dr
            fz = (z - z0) / dz
            ir = int(math.floor(fr))
            iz = int(math.floor(fz))
            if ir >= 0 and ir < nr - 1 and iz >= 0 and iz < nz - 1:
                wr1 = fr - ir
                wz1 = fz - iz
                wr0 = 1.0 - wr1
                wz0 = 1.0 - wz1
                if nphi == 1:
                    nplane = 1
                    ip0 = 0
                    ip1 = 0
                    wp0 = 1.0
                    wp1 = 0.0
                else:
                    nplane = 2
                    phi = math.atan2(y, x) % two_pi
                    fp = phi / dphi
                    ip0 = int(math.floor(fp))
                    wp1 = fp - ip0
                    ip0 = ip0 % nphi
                    ip1 = (ip0 + 1) % nphi
                    wp0 = 1.0 - wp1
                for q in range(nplane):
                    ip = ip0 if q == 0 else ip1
                    wp = wp0 if q == 0 else wp1
                    for corner in range(4):
                        ii = ir + (corner & 1)
                        kk = iz + (corner >> 1)
                        wr = wr1 if (corner & 1) else wr0
                        wz = wz1 if (corner >> 1) else wz0
                        w = wr * wz
                        if nphi > 1:
                            w = w * wp
                        g0 += w * fields[ii, ip, kk, 0]
                        g1 += w * fields[ii, ip, kk, 1]
                        g2 += w * fields[ii, ip, kk, 2]
                        g3 += w * fields[ii, ip, kk, 3]
                        g4 += w * fields[ii, ip, kk, 4]
                        g5 += w * fields[ii, ip, kk, 5]

            # cylindrical -> Cartesian
            ex = g0 * cp - g1 * sp
            ey = g0 * sp + g1 * cp
            ez = g2
            bx = g3 * cp - g4 * sp
            by = g3 * sp + g4 * cp
            bz = g5

            # ---- analytic background field
            rmag = max(math.sqrt(x * x + y * y + z * z), 1e-3)
            mdotr = m_vec[0] * x + m_vec[1] * y + m_vec[2] * z
            f1 = mdotr / rmag ** 5
            f3 = rmag ** 3
            bx += 3.0 * x * f1 - m_vec[0] / f3
            by += 3.0 * y * f1 - m_vec[1] / f3
            bz += 3.0 * z * f1 - m_vec[2] / f3
            bz += ubz
            if outer_bz != 0.0:
                bz += outer_bz * 0.5 * (1.0 + math.tanh((r - outer_r0) / width))
            if tor_b != 0.0 and r > 0.0:
                if r < r_core:
                    bphi = tor_b * (r / r_core)
                else:
                    bphi = tor_b * (r_core / r)
                bx += -bphi * sp
                by += bphi * cp

            # ---- corotation electric field  E = -(omega x r) x B
            if with_coro:
                ox = omega[1] * z - omega[2] * y
                oy = omega[2] * x - omega[0] * z
                oz = omega[0] * y - omega[1] * x
                ex += -(oy * bz - oz * by)
                ey += -(oz * bx - ox * bz)
                ez += -(ox * by - oy * bx)

            e_out[p, 0] = ex
            e_out[p, 1] = ey
            e_out[p, 2] = ez
            b_out[p, 0] = bx
            b_out[p, 1] = by
            b_out[p, 2] = bz

            # ---- Boris step
            vx = vel[p, 0]
            vy = vel[p, 1]
            vz = vel[p, 2]
            hk = q_over_m * (0.5 * dt)
            hx = hk * ex
            hy = hk * ey
            hz = hk * ez
            vmx = vx + hx
            vmy = vy + hy
            vmz = vz + hz
            tx = hk * bx
            ty = hk * by
            tz = hk * bz
            t2 = tx * tx + ty * ty + tz * tz
            fac = 2.0 / (1.0 + t2)
            sx = tx * fac
            sy = ty * fac
            sz = tz * fac
            vpx = vmx + (vmy * tz - vmz * ty)
            vpy = vmy + (vmz * tx - vmx * tz)
            vpz = vmz + (vmx * ty - vmy * tx)
            vqx = vmx + (vpy * sz - vpz * sy)
            vqy = vmy + (vpz * sx - vpx * sz)
            vqz = vmz + (vpx * sy - vpy * sx)
            vnx = vqx + hx
            vny = vqy + hy
            vnz = vqz + hz
            v_new[p, 0] = vnx
            v_new[p, 1] = vny
            v_new[p, 2] = vnz
            x_new[p, 0] = x + vnx * dt
            x_new[p, 1] = y + vny * dt
            x_new[p, 2] = z + vnz * dt


def num_chunks() -> int:
    """Number of private deposit grids (= Numba worker threads)."""
    return int(nb.get_num_threads()) if HAVE_NUMBA else 1


def deposit(pos, vel, r0, z0, dr, dz, nr, nphi, nz, buf):
    """Return the (4, ncell) sums [count, ur, uphi, uz] of an (nr, nphi, nz) grid."""
    return _deposit_kernel(pos, vel, r0, z0, dr, dz, nr, nphi, nz, buf)


def push(
    pos, vel, fields, r0, z0, dr, dz, nr, nphi, nz,
    m_vec, ubz, outer_bz, outer_r0, outer_w, tor_b, tor_r0,
    with_coro, omega, q_over_m, dt,
    e_out, b_out, x_new, v_new,
):
    _push_kernel(
        pos, vel, fields, r0, z0, dr, dz, nr, nphi, nz,
        m_vec, ubz, outer_bz, outer_r0, outer_w, tor_b, tor_r0,
        with_coro, omega, q_over_m, dt,
        e_out, b_out, x_new, v_new,
    )
