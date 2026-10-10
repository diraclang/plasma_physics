# Mode-resolved tearing-mode formulation

This project keeps the x-direction on a real spatial grid and resolves the periodic y,z dependence with a small retained set of Fourier modes. The code therefore solves a mixed representation:

- x: real finite-difference grid
- y,z: explicit Fourier harmonics $(k_y, k_z)$

The governing equations in this representation are:

## 1) Charge and current deposition

For a retained mode $(k_y, k_z)$, the Fourier coefficients of the particle moments are (macroparticle weight $w$, CIC shape $S$ in $x$, $V_\perp = L_y L_z$)

$$
\rho_k(x_i) = \frac{1}{dx\,V_\perp}\sum_j q_j w_j\, S(x_i-x_j)\, e^{-i(k_y y_j + k_z z_j)}
$$

$$
J_k(x_i) = \frac{1}{dx\,V_\perp}\sum_j q_j w_j\, \mathbf v_j\, S(x_i-x_j)\, e^{-i(k_y y_j + k_z z_j)}
$$

with $J$ deposited at the mid-step position with $\mathbf v^{n+1/2}$. A real field is $F=F_{eq}(x)+2\,\Re\sum_{k\in\text{half space}}F_k(x)e^{i(k_yy+k_zz)}$; modes $(m,n)$ and $(-m,-n)$ are the same real mode and are folded together. The $(0,0)$ equilibrium $\mathbf B_{eq}=(0,B_g,B_z(x))$ is a static external field that does no work.

## 2) Longitudinal electric field

The longitudinal field is derived from a scalar potential:

$$
E_L = -\nabla \phi
$$

and for each mode,

$$
E_{Lx,k}(x) = -\frac{\partial \phi_k}{\partial x}, \qquad
E_{Ly,k}(x) = -i k_y \phi_k(x), \qquad
E_{Lz,k}(x) = -i k_z \phi_k(x)
$$

with

$$
\frac{\partial^2 \phi_k}{\partial x^2} - (k_y^2 + k_z^2)\phi_k
= -\frac{\rho_k(x)}{\epsilon_0}
$$

This is the Helmholtz/Coulomb-split Poisson solve used in the code. For energy conservation the code uses the compact 3-point Laplacian for $\phi_k$ and takes the force as the exact gradient of the CIC-interpolated potential (not an interpolated central-difference $E$); then the longitudinal field energy is exactly $\tfrac12\int\rho\phi$. (Using interpolated central-difference $E$ gave a 4-5% energy gain independent of $dt$.)

## 3) Transverse field evolution

The transverse field satisfies the Maxwell curl equations, mode by mode:

$$
\frac{\partial B_k}{\partial t} = -\nabla \times E_{T,k}
$$

$$
\frac{\partial E_{T,k}}{\partial t} = c^2 \nabla \times B_k - \frac{J_{T,k}}{\epsilon_0}
$$

with the curl in the mixed x-grid/Fourier representation

$$
(\nabla \times F)_x = i k_y F_z - i k_z F_y
$$

$$
(\nabla \times F)_y = i k_z F_x - \frac{\partial F_z}{\partial x}
$$

$$
(\nabla \times F)_z = \frac{\partial F_y}{\partial x} - i k_y F_x
$$

The project uses a direct leapfrog update in x for this step, with the current split into longitudinal and transverse parts before the update.

## 4) Field evaluation at particle positions

At a particle location $(x_i, y_i, z_i)$, the field is reconstructed by summing over all retained modes:

$$
E(x_i,y_i,z_i) = 2\,\Re\sum_{k \in \mathcal{K}} E_k(x_i)e^{i(k_y y_i + k_z z_i)}
$$

$$
B(x_i,y_i,z_i) = B_{eq}(x_i) + 2\,\Re\sum_{k \in \mathcal{K}} B_k(x_i)e^{i(k_y y_i + k_z z_i)}
$$

where $\mathcal K$ is the half-space mode set and $x$-interpolation uses the same CIC weights as the deposition. The Lorentz force is $q(\mathbf E+\mathbf v\times\mathbf B)$ (no $1/c$, because $\mu_0\epsilon_0c^2=1$ and $\mu_0=1$), so $\Omega_{ce}=eB/m_e$.

## 5) Code check list

To validate the implementation, the following checks should pass:

1. A purely mode-resolved analytic field reconstructs exactly when summed back to real space.
2. The Poisson solve reproduces a known analytic potential for a single Fourier harmonic.
3. The current projection leaves only the transverse piece in the Maxwell update.
4. A vacuum light-wave test propagates at the correct speed as a Maxwell solver check.
5. A full simulation run shows bounded total-energy drift when the particle-field loop is consistent.

## 6) Energy audit and timestep

Conserved quantity (perturbation fields only, since $B_{eq}$ does no work):

$$
W = \sum_s \tfrac12 m_s w_s\, \mathbf v^{n-1/2}\!\cdot\mathbf v^{n+1/2} + \tfrac12\!\int\!\rho\phi + \tfrac{\epsilon_0}{2}\!\int\!|E_T|^2 + \tfrac{1}{2\mu_0}\!\int\! \mathbf B^{n-1/2}\!\cdot\mathbf B^{n+1/2}
$$

where field integrals are $2V_\perp\,dx\sum_{\text{half-space}}|\cdot|^2$. `run_coupled_loop()` returns $W_K, W_E, W_B, W_{total}$. The check is only meaningful when $W_E+W_B$ is a substantial fraction of the exchanged energy, so the check scripts use a seed amplitude and particle noise large enough to make that true.

Timestep (`stable_dt`): $\Omega_{ce}^{max}dt\le0.1$, $\omega_{pe}^{max}dt\le0.2$, $c\,dt/dx\le0.5$. For $B_0=B_g=4$, $e=0.4$ this gives $dt\approx0.044$ (not $10^{-5}$).

Measured drift (10 half-space modes from $m\in[-3,3]$, $n\in[-1,1]$, $\Delta x=\lambda_D$, 100 ppc): about $10^{-4}$ for $t\le22$, about $10^{-3}$ at $t\approx44$. Residual error is independent of $dt$, falls ~4.5x for 4x more particles, and comes from non-exact charge conservation of the mid-point current deposit (an Esirkepov-type deposit would remove it).

This file and the helper module `src/mode_gather.py` are intended to make those checks explicit and easy to audit while the full coupled time-stepping loop is assembled.
