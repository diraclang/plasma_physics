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

## 4) Species charge convention

The particle charge does not need to be stored as an arbitrary per-particle array in the same way as position or velocity. We only have two species:

- ions with charge `+e`
- electrons with charge `-e`

So a compact representation is:

```python
species_charge = {
    "ion": +e,
    "electron": -e,
}
```

and for each particle `p` we can write

```python
q[p] = species_charge[species[p]]
```

This makes the code simpler and matches the physical model exactly.

## 5) MLX kernel sketch for particle-parallel execution

The MLX implementation should be organized around particle blocks and mode reductions. The main idea is to keep the expensive vector work in array operations, not in Python loops over particles.

### 4.1 Data layout

Store the particle arrays as contiguous MLX arrays:

```python
x      : [N]   # particle positions in x
y      : [N]
z      : [N]
vx     : [N]
vy     : [N]
vz     : [N]
q      : [N]
w      : [N]
```

and the retained mode bank as arrays of wave numbers:

```python
kx_modes : [M]
ky_modes : [M]
kz_modes : [M]
```

For each particle, define a local phase cache for the retained modes:

```python
phase_x[p, ikx] = exp(1j * kx_modes[ikx] * x[p])
phase_y[p, iky] = exp(1j * ky_modes[iky] * y[p])
phase_z[p, ikz] = exp(1j * kz_modes[ikz] * z[p])
```

with the higher harmonics generated via repeated multiplication instead of repeated `exp()` calls.

### 4.2 Particle deposition kernel

A single MLX kernel for one particle block does:

```python
# pseudocode
for p in block_particles:
    for ik in modes:
        kx, ky, kz = mode_bank[ik]
        phase = phase_x[p, kx] * phase_y[p, ky] * phase_z[p, kz]
        rho[ik] += q[p] * w[p] * conj(phase)
        Jx[ik] += q[p] * w[p] * vx[p] * conj(phase)
        Jy[ik] += q[p] * w[p] * vy[p] * conj(phase)
        Jz[ik] += q[p] * w[p] * vz[p] * conj(phase)
```

This is implemented as a vectorized MLX reduction over the particle axis:

```python
rho = mx.sum(q * w * conj(phase), axis=0)
Jx  = mx.sum(q * w * vx * conj(phase), axis=0)
Jy  = mx.sum(q * w * vy * conj(phase), axis=0)
Jz  = mx.sum(q * w * vz * conj(phase), axis=0)
```

where `phase` is shaped like `[N_particles, N_modes]`.

### 4.3 Maxwell update kernel

After deposition, the Fourier-space EM fields are advanced using the mode data:

```python
for ik in modes:
    k = (kx[ik], ky[ik], kz[ik])
    # algebraic Maxwell update in spectral space
    B_hat[ik] = B_hat[ik] - dt * curl(E_hat[ik])
    E_hat[ik] = E_hat[ik] + dt * (c**2 * curl(B_hat[ik]) - J_hat[ik] / eps0)
```

In MLX this is done as array operations over the mode axis, without any Python loop over the whole mode bank if the bank is modest in size.

### 4.4 Field evaluation at particle positions

After advancing the field coefficients, reconstruct the field at each particle location by another spectral sum:

```python
for p in particles:
    Ex[p] = sum_ik E_hat[ik] * phase[p, ik]
    Ey[p] = sum_ik E_hat_y[ik] * phase[p, ik]
    Ez[p] = sum_ik E_hat_z[ik] * phase[p, ik]
    Bx[p] = sum_ik B_hat_x[ik] * phase[p, ik]
    By[p] = sum_ik B_hat_y[ik] * phase[p, ik]
    Bz[p] = sum_ik B_hat_z[ik] * phase[p, ik]
```

Vectorized MLX form:

```python
phase = phase_x[:, :, None] * phase_y[:, :, None] * phase_z[:, :, None]
# shape [N_particles, N_modes]
E = phase @ E_hat
B = phase @ B_hat
```

where `E_hat` is the mode-valued electric field array. This is a dense matrix-vector product, but with `N_particles` much larger than the number of retained modes, and it is highly parallelizable on GPU.

### 4.5 Particle push kernel

Then the particle force is evaluated as

$$
\mathbf{F}_p = q_p\left(\mathbf{E}_p + \mathbf{v}_p\times\mathbf{B}_p\right)
$$

and the push is applied:

```python
ax = (q / m) * (Ex + vy * Bz - vz * By)
ay = (q / m) * (Ey + vz * Bx - vx * Bz)
az = (q / m) * (Ez + vx * By - vy * Bx)

vx = vx + dt * ax
vy = vy + dt * ay
vz = vz + dt * az
x  = x  + dt * vx
y  = y  + dt * vy
z  = z  + dt * vz
```

This is all done as vectorized MLX array updates, not by Python loops over particles.

### 4.6 Typical MLX execution pattern

The full step becomes:

```python
# 1. evaluate phase cache for all particles and modes
phase = build_phase_cache(x, y, z, mode_bank)

# 2. deposit density and current
rho, J = deposit_modes(particles, phase, mode_bank)

# 3. Maxwell advance in Fourier space
E_hat, B_hat = advance_maxwell(rho, J, E_hat, B_hat, dt)

# 4. field at particles
E_part, B_part = evaluate_particle_fields(phase, E_hat, B_hat)

# 5. push particles
x, y, z, vx, vy, vz = push_particles(x, y, z, vx, vy, vz,
                                     E_part, B_part, q, m, dt)
```

This makes the MLX port map directly onto the same algorithmic structure as the direct spectral code.

### 4.7 Why this is a good MLX fit

This pattern is good for MLX because:

- particle arrays are large and uniform,
- the mode bank is relatively small,
- the dominant work is array-based reduction and matrix multiplication,
- the phase-factor reuse reduces the expensive transcendental work,
- the GPU work is naturally parallel across particles and modes.

The important thing is that the code should remain batch-array based. We should avoid Python `for` loops over each particle unless absolutely necessary.

---

## Summary

The final architecture is:

- full 3D direct spectral representation,
- particle arrays in MLX,
- mode-by-mode density/current deposition,
- mode-space Maxwell advance,
- field reconstruction at particle positions from the same phase factors,
- vectorized particle push,
- cached harmonic phase factors to remove repeated expensive `exp()` calls.

This is the correct MLX implementation strategy for the new code.
