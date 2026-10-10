"""
Per-mode longitudinal field solve: Poisson's equation for the scalar
potential phi, one independent 1D boundary-value problem (periodic in x) per
retained Fourier harmonic (k_y, k_z):

    d^2 phi_k / dx^2 - (k_y^2 + k_z^2) * phi_k = -rho_k(x) / eps_0

rho_k(x) and phi_k(x) are COMPLEX amplitudes (Fourier coefficients in y,z);
x stays a real grid direction, discretized with a standard periodic
second-order finite-difference stencil:

    (phi[i-1] - 2*phi[i] + phi[i+1]) / dx^2 - kperp2 * phi[i] = -rho[i] / eps_0

This gives a periodic (cyclic) tridiagonal linear system A @ phi = b. Since
the matrix A depends only on kperp2 = k_y^2 + k_z^2 (the SAME for the whole
simulation, for a given mode), we factor it ONCE per mode at setup time and
reuse that factorization every timestep (fast back-substitution only).

Special case kperp2 == 0 (the m=n=0 "mode", i.e. the x-only mean/background):
the periodic 1D Poisson problem is singular (constant functions are in the
null space) and only solvable if the mean of rho is zero (charge neutrality
on average) -- exactly satisfied here since ion and electron equilibrium
densities are identical (quasineutral). We handle this by solving on the
orthogonal complement of the constant mode (pin phi's spatial mean to zero,
which is just a gauge choice -- E = -dphi/dx is unaffected by an additive
constant in phi).
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class ModePoissonSolver:
    """Precomputed, factored periodic-tridiagonal Poisson operator for one
    specific kperp2 value. Reused every timestep for that mode."""
    nx: int
    dx: float
    kperp2: float
    eps_0: float
    is_degenerate: bool   # True iff kperp2 == 0 (needs zero-mean projection)
    lu_piv: tuple          # output of scipy.linalg.lu_factor, for non-degenerate case
    A_reduced_lu_piv: tuple = None  # factorization used in the degenerate (kperp2=0) case

    def solve(self, rho_k: np.ndarray) -> np.ndarray:
        """Solve for phi_k(x) given rho_k(x) (both length-nx complex arrays)."""
        rhs = -rho_k / self.eps_0 * self.dx ** 2
        if not self.is_degenerate:
            return lu_solve(self.lu_piv, rhs)
        # kperp2 == 0: project out the mean (compatibility condition), then
        # solve on the (nx-1)-dim reduced system with phi[0] pinned to 0.
        rhs = rhs - np.mean(rhs)
        rhs_reduced = rhs[1:]
        phi_reduced = lu_solve(self.A_reduced_lu_piv, rhs_reduced)
        phi = np.concatenate(([0.0 + 0.0j], phi_reduced))
        return phi - np.mean(phi)  # re-center for a clean, deterministic gauge


def build_mode_poisson_solver(nx: int, dx: float, kperp2: float, eps_0: float) -> ModePoissonSolver:
    """Build and factor the periodic tridiagonal Poisson operator for one mode."""
    diag_val = -(2.0 + kperp2 * dx ** 2)
    is_degenerate = (kperp2 == 0.0)

    if not is_degenerate:
        A = np.zeros((nx, nx), dtype=float)
        for i in range(nx):
            A[i, i] = diag_val
            A[i, (i - 1) % nx] += 1.0
            A[i, (i + 1) % nx] += 1.0
        lu_piv = lu_factor(A)
        return ModePoissonSolver(nx=nx, dx=dx, kperp2=kperp2, eps_0=eps_0,
                                  is_degenerate=False, lu_piv=lu_piv)
    else:
        # Degenerate case: drop row/col 0 and pin phi[0]=0 to remove the null
        # space (shifting by the mean is handled in .solve()).
        n_red = nx - 1
        A = np.zeros((n_red, n_red), dtype=float)
        for ii in range(n_red):
            i = ii + 1  # actual grid index, 1..nx-1
            A[ii, ii] = diag_val
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            if im1 != 0:
                A[ii, im1 - 1] += 1.0
            if ip1 != 0:
                A[ii, ip1 - 1] += 1.0
        A_reduced_lu_piv = lu_factor(A)
        return ModePoissonSolver(nx=nx, dx=dx, kperp2=kperp2, eps_0=eps_0,
                                  is_degenerate=True, lu_piv=None,
                                  A_reduced_lu_piv=A_reduced_lu_piv)
