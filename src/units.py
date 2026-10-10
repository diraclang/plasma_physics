"""
Physical units and normalization convention for this code.

This matches the original 1993 code's convention directly: a small set of
FIXED base units (independent choices), with everything else derived.

Fixed base units:
    lambda_De = 1      (electron Debye length -- length unit)
    mu_0      = 1      (vacuum permeability)
    n_0       = 1      (reference/peak sheet density)
    m_e       = 1      (electron mass -- mass unit)

Direct physics inputs (from SimulationConfig):
    mass_ratio = m_i / m_e
    c          = speed of light (a real, finite number in these units --
                 light waves are genuinely resolved by this code)
    B0         = reversing field peak amplitude (Bz)
    Bg         = uniform guide field amplitude (By)
    T_i_over_T_e = ion/electron temperature ratio at the sheet center

Derived (see derive_units()):
    eps_0 = 1 / (mu_0 * c^2)                      (from mu_0 eps_0 c^2 = 1)
    m_i   = mass_ratio * m_e
    T_e   = set so that lambda_De = 1 exactly, i.e. lambda_De^2 =
            eps_0 * T_e / (n_0 * e^2); combined with the definition
            omega_pe = sqrt(n_0 e^2 / (eps_0 m_e)) this gives the pair
            (e, T_e) uniquely once we also fix V_Te (electron thermal speed)
            as a physics input -- see derive_units() for the exact algebra.
    T_i   = T_e * T_i_over_T_e
    omega_pe, omega_pi, d_i, Omega_ci, Omega_ce, v_A -- all standard formulas.

Worked numeric example (the 1993 parameters): mass_ratio=4, c=5, B0=Bg=4,
T_i_over_T_e=1, V_Te=2, V_Ti=1  =>  eps_0=0.04, e=0.4, omega_pe=2, omega_pi=1,
d_i=5 (lambda_De units), Omega_ci=0.4, Omega_ce=1.6, v_A=2.  (Verified:
v_A = B0/sqrt(mu_0 n0 m_i) = 4/sqrt(1*1*4) = 2, matches Omega_ci*d_i = 2.)
The Lorentz force is q(E + v x B) with mu_0=1 (SI-like), so Omega = e B / m
with NO 1/c.  (An earlier version divided by c, giving 0.08 / 0.32; wrong for
this convention.)
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DerivedUnits:
    lambda_De: float
    mu_0: float
    eps_0: float
    n0: float
    m_e: float
    m_i: float
    e: float           # elementary charge magnitude; q_i=+e, q_e=-e
    q_i: float
    q_e: float
    c: float
    omega_pe: float
    omega_pi: float
    d_i: float
    Omega_ci: float
    Omega_ce: float
    v_A: float
    T_e: float
    T_i: float


def derive_units(mass_ratio: float, c: float, B0: float, V_Te: float,
                  T_i_over_T_e: float = 1.0, n0: float = 1.0,
                  mu_0: float = 1.0, lambda_De: float = 1.0,
                  m_e: float = 1.0) -> DerivedUnits:
    """Derive the full self-consistent set of plasma parameters from the
    fixed base units (lambda_De, mu_0, n0, m_e -- all =1 by convention) plus
    the direct physics inputs (mass_ratio, c, B0, V_Te, T_i_over_T_e).

    V_Te is the electron thermal speed, V_Te = sqrt(T_e / m_e); giving this
    directly (rather than a beta or a temperature) mirrors how the original
    1993 code's parameters were reported.
    """
    eps_0 = 1.0 / (mu_0 * c ** 2)
    m_i = mass_ratio * m_e
    T_e = m_e * V_Te ** 2
    T_i = T_e * T_i_over_T_e
    # lambda_De^2 = eps_0 * T_e / (n0 * e^2) = lambda_De^2  =>  solve for e
    e = math.sqrt(eps_0 * T_e / n0) / lambda_De
    q_i = e
    q_e = -e
    omega_pe = math.sqrt(n0 * e ** 2 / (eps_0 * m_e))
    omega_pi = omega_pe / math.sqrt(mass_ratio)
    d_i = c / omega_pi
    Omega_ci = e * B0 / m_i
    Omega_ce = e * B0 / m_e
    v_A = B0 / math.sqrt(mu_0 * n0 * m_i)

    return DerivedUnits(
        lambda_De=lambda_De, mu_0=mu_0, eps_0=eps_0, n0=n0, m_e=m_e, m_i=m_i,
        e=e, q_i=q_i, q_e=q_e, c=c, omega_pe=omega_pe, omega_pi=omega_pi,
        d_i=d_i, Omega_ci=Omega_ci, Omega_ce=Omega_ce, v_A=v_A,
        T_e=T_e, T_i=T_i,
    )
