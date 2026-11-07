# Executive Summary for Business Plan

## 1. Mission
Develop a dipole-based fusion concept through simulation-first physics and control, then de-risk toward hardware milestones that investors can evaluate.

## 2. Current Position
- The team has established stable dipole-only confinement behavior in long runs.
- Radial confinement is strong in the latest hold-at-100 runs, while vertical spreading remains the dominant issue.
- Time-ramped fields and continuation workflows are operational.
- Diagnostics and plotting now support reproducible meridian field and particle analysis.

## 3. What We Learned
- Baseline comparisons must keep initial conditions and parameters identical.
- Uniform bias field sign matters strongly. Wrong sign weakens equatorial confinement.
- Disk-safe workflows are essential for long runs and validation loops.
- The model supports a coherent normalized-to-SI mapping for realistic design discussion.

## 4. Physical Design Anchor (Agreed Baseline)
Design Point A:
- Ion species: deuterium ions
- Blob radius target: about 1 meter
- Equatorial field at blob: about 5 tesla
- Temperature target: Ti and Te around 10 keV
- Beta target: around 0.05

Implications:
- Required dipole moment is large but plausible for superconducting magnet programs.
- The concept is likely prototype-feasible with current magnet technology, but reactor-grade deployment remains a harder engineering phase.

## 5. Feasibility View
- 5 tesla class operation is feasible with current superconducting technology.
- The main challenge is not only field generation, but structural stress, cryogenics, quench protection, shielding, and maintainability.
- A practical path is staged: simulation validation, physics prototype, then engineering scale-up.

## 6. Cost Framing (Materials-Only, Order of Magnitude)
- Prototype-class hardware: tens of millions of USD.
- Robust high-field system: roughly 100 million class.
- Reactor-relevant materials stack: can approach high hundreds of millions to billion class.

## 7. Investment Thesis
Investors usually do not fund simulation alone. They fund simulation that reduces hardware risk with measurable outcomes.

The investable narrative:
- Build a control-capable plasma physics engine.
- Demonstrate that control reduces escape/spread for fixed peak field.
- Translate that reduction into lower magnet and system cost risk.
- Validate predictions in staged experiments.

## 8. Strategic Story: Control as the Core Differentiator
The strongest near-term story is closed-loop plasma control:
- Treat dipole moment and optional bias as control inputs over time.
- Include physically consistent induced electric field effects for changing magnetic fields.
- Benchmark open-loop versus feedback control using objective metrics.

Primary metrics:
- Escape fraction
- Radial and vertical spread growth
- Energy growth bounds
- Control effort and actuator smoothness

## 9. Validation Without a Full Reactor
Use layered validation:
1. Internal consistency and convergence tests.
2. Recovery of known analytic limits and invariants.
3. Reproduction of known dipole experiment trends from literature.
4. Falsifiable predictions for a smaller hardware test.

This turns the program from simulation-only to decision-grade physics and control.

## 10. Near-Term Milestones
1. Finish current long continuation and compare settling trends.
2. Add control API and baseline PID controller.
3. Add Faraday-consistent induced field term for imposed magnetic ramps.
4. Run A/B comparisons:
   - Open-loop hold
   - Hand-tuned schedule
   - Closed-loop control
5. Publish a milestone report with uncertainty bars and hardware impact.

## 11. Bottom Line
The path is credible if the team focuses on control-driven de-risking with measurable milestones.

Success condition:
- Show that simulation-guided control can reduce confinement losses and engineering requirements before full-scale hardware.

That is the bridge from strong technical work to an investable program.