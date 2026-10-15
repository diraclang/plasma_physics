# Parallel Computing Sandbox

This directory is intentionally kept separate from the current plasma-physics source tree.
It is a clean workspace for the new all-spectral, MLX-based solver design that does not depend on the legacy x-grid code.

## Goals

- keep the new implementation independent from the current plasma-physics source
- use a fully spectral treatment in all three spatial dimensions
- keep the mode bank explicit and controlled
- use MLX kernels for batched work on Apple Silicon
- preserve a clean separation between research, architecture, and production code

## Layout

- `src/` : new solver implementation
- `notebooks/` : analysis notebooks when needed
- `runs/` : local run outputs and checkpoints

## Current direction

The core solver should use:

- Fourier modes in x, y, z
- no x-grid in the main time-stepping loop
- deposition into retained modes
- mode-wise field updates
- batched reductions across retained mode banks
- MLX kernels for expensive vectorized work

## First particle-parallel prototype

This folder now includes a simpler test problem that is intentionally easier to reason about and faster to run on GPU hardware:

- a uniform magnetic field B = (0, 0, B0)
- particle motion updated with a Boris-style push
- particles processed in blocks to mimic GPU parallelism
- MLX arrays available when the environment provides them

Run it with the device pinned explicitly to GPU:

```bash
cd /Users/zhiwang/diraclang/plasma-physics/parallel-computing
PYTHONPATH=. python -m src.simple_magnetic_solver --steps 10 --particles 4096 --B0 1.0 --block-size 256 --device gpu
```

This forces MLX to use the Metal GPU by calling `mx.set_default_device(mx.gpu)` before the particle arrays are allocated. The script also prints the chosen device in its output so you can verify the runtime is not silently staying on the host.

## Minimal local usage

```bash
cd /Users/zhiwang/diraclang/plasma-physics/parallel-computing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m src.simple_magnetic_solver --help
```

The code in this folder is a clean experimental branch and should not be imported from the older solver unless explicitly desired.
