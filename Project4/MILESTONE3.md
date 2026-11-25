# Milestone 3 — Bridge vibration and linear buckling

## Scope

This checkpoint adds the two eigenproblems needed for R4 and R5 while preserving the verified
3D frame/static implementation from Milestones 1–2.

## Vibration (R4)

The supported bridge solves

\[
K\phi = \omega^2 M\phi, \qquad f=\frac{\omega}{2\pi}.
\]

`M` contains the consistent 3D frame-element steel mass and the exact D2 deck mass lumped to
the five longitudinal deck support lines. Pedestrian mass is intentionally excluded, matching
the specification.

Initial design (`r=50 mm`, `H=2.0 m`):

| mode | frequency (Hz) | qualitative character |
|---:|---:|---|
| 1 | 4.778009 | predominantly lateral sway |
| 2 | 9.531065 | predominantly vertical |
| 3 | 10.201083 | lateral/mixed |
| 4 | 10.668553 | lateral/mixed |
| 5 | 15.050114 | predominantly vertical |
| 6 | 15.644295 | lateral/mixed |

Therefore the correct R4 value is **4.778009 Hz**, because the requirement uses the lowest
frequency of any mode, not the first vertical mode. It passes the 3 Hz limit.

## Linear buckling (R5 / S1)

The static design-load solution is solved first. For each FE element, the recovered axial force
is taken positive in tension and negative in compression. The geometric stiffness is assembled
with the same sign convention, so the stability equation is

\[
(K + \alpha K_G)\psi=0.
\]

For numerical robustness it is solved as the symmetric generalized eigenproblem

\[
(-K_G)\psi = \mu K\psi, \qquad \alpha=\frac{1}{\mu}.
\]

The largest positive `mu` therefore gives the smallest positive buckling factor.

Initial design results:

- critical buckling factor: **21.607746**;
- next factors: 21.799646, 21.810471, 21.837768, 22.044619, 22.085645;
- element axial-force range under the design load: **-128.964 to +98.259 kN**;
- R5 limit `alpha_cr >= 3`: **PASS**.

The initial critical factor is also close to the instructor prototype's reported 20.9, despite
the independently constructed member layout.

## Verification added

`verification/run_eigen_bridge_checks.py` exercises the production eigen routines against the
analytical V3 and V4 benchmarks.

### V3 beam frequency

The first-frequency relative error decreases from 10.99% with one element to about
`1.03e-6` with 16 elements.

### V4 Euler column

The Euler buckling relative error decreases from 21.59% with one element to about
`3.28e-5` with 8 elements.

### Bridge eigen residuals

- vibration generalized-eigenproblem relative residual: approximately `4e-12`;
- buckling generalized-eigenproblem relative residual: approximately `4e-14`.

All previous cantilever, rotation-invariance, distributed-load/stress-recovery, rigid-body,
reaction-balance, D2 plate, and full bridge static checks continue to pass.

## Files added

- `src/eigen.py` — vibration, axial-force extraction, geometric-stiffness assembly, buckling;
- `scripts/run_bridge_eigen.py` — initial bridge R4/R5 analysis and JSON output;
- `verification/run_eigen_bridge_checks.py` — V3/V4 convergence plus bridge eigen residuals;
- `assets/milestone3_results.json` — machine-readable checkpoint results.

## Next milestone

Implement the response/gradient layer and V6 Taylor tests. The sensible sequence is mass and
static responses first, then vibration eigenvalue sensitivity, and finally the S1 buckling
sensitivity because `K_G` depends on the static displacement state and therefore needs the
additional adjoint-through-state term.
