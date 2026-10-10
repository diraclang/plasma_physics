"""
Per-mode current projection: split the deposited current J_k(x) (complex
Fourier amplitude, per retained (k_y, k_z) harmonic) into its longitudinal
(curl-free) and transverse (divergence-free) parts.

WHY THIS IS NEEDED: this code solves the longitudinal electric field E_L via
a direct Poisson solve (fields_longitudinal.py) from the instantaneous charge
density, rather than by evolving E_L dynamically via Ampere's law. If the
transverse-field Maxwell leapfrog were driven by the FULL current J (not just
its transverse part J_T), it would create a SECOND, redundant response to the
same charge dynamics already handled by the Poisson solve -- double-counting
the longitudinal physics. Projecting out J_L before using the current in the
transverse Maxwell equation avoids this.

Method (Helmholtz decomposition in the mixed x-grid / (k_y,k_z)-Fourier
representation): find a scalar psi_k(x) such that

    d^2 psi_k/dx^2 - kperp2 * psi_k = div(J_k)
                                     = dJx_k/dx + i*k_y*Jy_k + i*k_z*Jz_k

then

    J_L_k = grad(psi_k) = ( dpsi_k/dx ,  i*k_y*psi_k ,  i*k_z*psi_k )
    J_T_k = J_k - J_L_k

This is the SAME elliptic operator as the longitudinal Poisson solve (same
matrix, same factorization can be reused), just with a different right-hand
side -- so we reuse ModePoissonSolver directly.
"""
import numpy as np

from fields_longitudinal import ModePoissonSolver


def project_transverse_current(Jx_k: np.ndarray, Jy_k: np.ndarray, Jz_k: np.ndarray,
                                ky: float, kz: float, dx: float,
                                poisson_solver: ModePoissonSolver):
    """Return (JTx_k, JTy_k, JTz_k), the transverse (divergence-free) part of
    the current, given the full deposited current components for one mode.
    `poisson_solver` must have been built with the SAME kperp2 = ky^2+kz^2 as
    this mode (its eps_0 scaling is irrelverant here -- see below)."""
    nx = Jx_k.shape[0]
    dJx_dx = _central_diff_periodic(Jx_k, dx)
    div_J = dJx_dx + 1j * ky * Jy_k + 1j * kz * Jz_k

    # Reuse the Poisson solver machinery: it solves
    #   psi'' - kperp2*psi = -source/eps_0
    # we want
    #   psi'' - kperp2*psi = div_J
    # so pass source = -div_J * eps_0 to cancel the solver's internal /eps_0.
    psi_k = poisson_solver.solve(-div_J * poisson_solver.eps_0)

    dpsi_dx = _central_diff_periodic(psi_k, dx)
    JLx_k = dpsi_dx
    JLy_k = 1j * ky * psi_k
    JLz_k = 1j * kz * psi_k

    JTx_k = Jx_k - JLx_k
    JTy_k = Jy_k - JLy_k
    JTz_k = Jz_k - JLz_k
    return JTx_k, JTy_k, JTz_k


def _central_diff_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central difference, periodic boundary conditions."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)
