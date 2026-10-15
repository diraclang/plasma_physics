# Direct 3D spectral EM/PIC formulation

This code is a full three-dimensional spectral particle-in-cell solver. We do not use a real-space x-grid at all. All three coordinates are represented in Fourier space, and the entire EM/PIC update is done mode-by-mode. We never do a numerical FFT or a grid-based gather/scatter step.

## 1) Equations and algorithm

### 1.1 Spectral basis

The field and moments are expanded in the retained Fourier basis:

$$
\phi(\mathbf{x}) = \sum_{\mathbf{k}\in\mathcal{K}} \hat\phi_{\mathbf{k}} \, e^{i\mathbf{k}\cdot\mathbf{x}}
$$

with

$$
\mathbf{k} = (k_x, k_y, k_z).
$$

The code keeps only the selected mode bank $\mathcal{K}$, with the mean/mode-0 sector handled explicitly when needed.

### 1.2 Particle equations of motion

For each particle $p$,

$$
\frac{d\mathbf{x}_p}{dt} = \mathbf{v}_p,
\qquad
\frac{d\mathbf{v}_p}{dt} = \frac{q_p}{m_p}\left(\mathbf{E}(\mathbf{x}_p) + \mathbf{v}_p\times\mathbf{B}(\mathbf{x}_p)\right).
$$

The field at the particle position is assembled directly from the retained Fourier modes:

$$
\mathbf{E}(\mathbf{x}_p) = \sum_{\mathbf{k}\in\mathcal{K}} \hat{\mathbf{E}}_{\mathbf{k}} \, e^{i\mathbf{k}\cdot\mathbf{x}_p},
$$

$$
\mathbf{B}(\mathbf{x}_p) = \mathbf{B}_{eq}(\mathbf{x}_p) + \sum_{\mathbf{k}\in\mathcal{K}} \hat{\mathbf{B}}_{\mathbf{k}} \, e^{i\mathbf{k}\cdot\mathbf{x}_p}.
$$

This is the field evaluation used in the push. There is no x-grid, no x-space interpolation, and no numerical FFT.

### 1.3 Density and current accumulation in Fourier modes

For particle weight $w_p$, charge $q_p$, and velocity $\mathbf{v}_p$, the mode-resolved charge and current are accumulated directly as

$$
\hat\rho_{\mathbf{k}} = \sum_p q_p w_p \, e^{-i\mathbf{k}\cdot\mathbf{x}_p},
$$

$$
\hat{\mathbf{J}}_{\mathbf{k}} = \sum_p q_p w_p \, \mathbf{v}_p \, e^{-i\mathbf{k}\cdot\mathbf{x}_p}.
$$

This is the fundamental particle-to-mode deposition. It happens directly in the retained spectral basis and does not require any real-space FFT or gather/scatter step.

### 1.4 Maxwell equations in Fourier space

For each retained mode $\mathbf{k}$, the EM update is

$$
\frac{\partial \hat{\mathbf{B}}_{\mathbf{k}}}{\partial t} = -\nabla\times\hat{\mathbf{E}}_{\mathbf{k}},
$$

$$
\frac{\partial \hat{\mathbf{E}}_{\mathbf{k}}}{\partial t} = c^2 \nabla\times\hat{\mathbf{B}}_{\mathbf{k}} - \frac{1}{\epsilon_0}\hat{\mathbf{J}}_{\mathbf{k}}.
$$

Because all coordinates are spectral, the curl operator becomes purely a multiplication by $i\mathbf{k}$ and the appropriate tensorial factors. The update is therefore done mode-by-mode in Fourier space.

### 1.5 Time-stepping loop

The loop is:

1. evaluate the field at particle positions from the current spectral mode coefficients,
2. push particle positions and velocities,
3. deposit charge/current directly into the retained Fourier modes,
4. advance the EM fields in Fourier space,
5. repeat.

This is the direct spectral EM/PIC algorithm.

---

## 2) GPU parallelization by particle groups

The natural parallelization is particle-centric.

### 2.1 Particle blocks

- Split the particles into blocks or tiles.
- One GPU block handles a chunk of particles.
- Each thread processes one particle or a short vector of particles.

### 2.2 Local particle work per GPU block

For each particle in the block:

- read position $\mathbf{x}_p$ and velocity $\mathbf{v}_p$,
- compute the mode weights $e^{i\mathbf{k}\cdot\mathbf{x}_p}$ for the retained mode bank,
- accumulate local contributions to the density and current for each mode,
- compute a local field value at that particle if needed for the push.

### 2.3 Reduction to global mode moments

Each block accumulates local spectral moments, and then the block results are reduced to the global mode arrays:

$$
\hat\rho_{\mathbf{k}} = \sum_{b} \hat\rho_{\mathbf{k}}^{(b)},
\qquad
\hat{\mathbf{J}}_{\mathbf{k}} = \sum_{b} \hat{\mathbf{J}}_{\mathbf{k}}^{(b)}.
$$

This is a reduction over only the retained mode bank, not over a full grid.

### 2.4 Maxwell advance on the mode data

Once the global mode moments are available, advance all retained EM Fourier coefficients on the GPU:

- update $\hat{\mathbf{E}}_{\mathbf{k}}$,
- update $\hat{\mathbf{B}}_{\mathbf{k}}$,
- keep the mean mode and equilibrium sector explicit when needed.

### 2.5 Particle push

Each particle block then reads the updated spectral field coefficients, evaluates the field at that particle position by spectral summation, and pushes the particle. The structure is therefore:

1. particle block computes local phase factors and mode deposition,
2. reduce to global mode moments,
3. advance EM in spectral space,
4. evaluate field at particle positions from the modal coefficients,
5. push particles.

This is the correct particle-parallel organization for a full spectral PIC code.

---

## 3) Saving cost on the phase factor $e^{i x}$

The main optimization is to avoid repeatedly calling `exp(i k x)` for every mode and every particle. We reuse the phase factor algebraically.

### 3.1 Fundamental phase reuse

Let

$$
\xi_x = e^{i x}, \qquad \xi_y = e^{i y}, \qquad \xi_z = e^{i z}.
$$

Then for integer wave numbers,

$$
 e^{i k_x x} = \xi_x^{k_x},
\qquad
 e^{i k_y y} = \xi_y^{k_y},
\qquad
 e^{i k_z z} = \xi_z^{k_z}.
$$

So instead of calling `exp()` for each mode, we compute the base phase once and generate higher powers by repeated multiplication:

$$
\xi_x^0 = 1,
\quad \xi_x^1 = \xi_x,
\quad \xi_x^m = \xi_x^{m-1}\xi_x.
$$

Similarly for $\xi_y$ and $\xi_z$.

### 3.2 Combined mode factors

For a mode $\mathbf{k}=(k_x,k_y,k_z)$,

$$
 e^{i\mathbf{k}\cdot\mathbf{x}} = e^{i k_x x} e^{i k_y y} e^{i k_z z}
 = \xi_x^{k_x} \xi_y^{k_y} \xi_z^{k_z}.
$$

This gives the exact spectral phase factor for each particle and retained mode with only a few multiplies after the base phase is known.

### 3.3 Particle-local phase cache

For each particle, we store:

```python
phase_x[0] = 1
phase_x[1] = exp(1j * x)
for mx in range(2, kx_max + 1):
    phase_x[mx] = phase_x[mx - 1] * phase_x[1]

phase_y[0] = 1
phase_y[1] = exp(1j * y)
for my in range(2, ky_max + 1):
    phase_y[my] = phase_y[my - 1] * phase_y[1]

phase_z[0] = 1
phase_z[1] = exp(1j * z)
for mz in range(2, kz_max + 1):
    phase_z[mz] = phase_z[mz - 1] * phase_z[1]
```

Then,

```python
mode_factor = phase_x[kx] * phase_y[ky] * phase_z[kz]
```

This is the key performance optimization.

### 3.4 Hermitian symmetry

For real-valued fields,

$$
\hat\phi_{-\mathbf{k}} = \hat\phi_{\mathbf{k}}^*.
$$

Thus we only compute the positive half of the mode bank and reconstruct the negative modes by conjugation where needed. This reduces both memory and arithmetic.

---

## Summary

The correct new implementation is a full three-dimensional spectral EM/PIC solver:

- no x-grid,
- all three dimensions in Fourier space,
- density/current accumulated directly in mode space,
- EM equations advanced in Fourier space,
- particle force evaluated from the mode coefficients at the particle positions,
- expensive `exp(i k x)` calls replaced by cached harmonic phase factors built from repeated multiplication.

This is the direct spectral formulation we are implementing.

## The design we are aiming for

The rewrite should be structured around:

- all physics represented in Fourier/mode space
- no x-grid dependence in the core solver
- direct mode evaluation for density/current contributions
- particle-parallel deposition and interpolation
- mode reductions done as GPU-friendly reductions
- MLX kernels for batched particle operations
- phase-factor reuse and compact mode-bank storage

This is the clean path to getting the performance and correctness aligned.
