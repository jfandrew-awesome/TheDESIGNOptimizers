# High-level design review before optimization

## Verdict
The implementation is internally consistent enough to proceed to optimization. The full verification suite was rerun from source after Milestone 4 and passes.

## Requirements traceability
- 3D frame solver: own Euler-Bernoulli implementation, sparse global assembly.
- Bridge: 15 m span, 3 m width, CHS t=r/10, 2 FE elements per physical member.
- D2 deck: deck stiffness excluded; pressure/self-weight distributed along five longitudinal support lines; deck mass included in vibration; 0.75 m support spacing passes the required strip check (37.02 MPa < 53.3 MPa).
- Supports: translational restraints applied to all deck nodes on each end line with x free on the roller end.
- R1/R2: end-force von Mises recovery; R2's 53.3 MPa equivalent Goodman limit governs R1.
- R3: all deck-node downward deflections checked.
- R4: lowest mode of any type used; the model's first mode is lateral at 4.778 Hz.
- R5: S1 state-dependent geometric stiffness and generalized buckling eigenproblem.
- R6: explicit member L/(0.673 r) constraint is included in the optimization driver.
- Shape variables: H and parabolic top-chord camber.

## Independent sanity checks
The independently chosen bridge geometry happens to reproduce the instructor prototype's initial static response closely (22.40 mm vs 22.2 mm deck deflection; 173.15 vs 173.0 MPa stress) and buckling factor reasonably closely (21.61 vs 20.9). These comparisons are treated only as sanity checks, not verification targets.

## Verification rerun
V1/V2 exact statics, V3 frequency convergence, V4 Euler buckling convergence, V5 rigid modes/global balance, distributed-load/stress recovery, bridge mass/equilibrium checks, eigen residuals, and V6 179-variable directional Taylor tests all pass.

## Caveats to preserve in the report
1. This is our bridge concept, not a clone of the instructor prototype. Its lowest mode is a 4.78 Hz lateral mode; this is valid and must not be replaced by the first vertical mode (~9.53 Hz).
2. KS aggregates are conservative smooth constraints. Final tables must report both aggregate values and the true stress/deflection/frequency/buckling extrema.
3. Shape partials are centered numerical partial derivatives of assembled operators at fixed state, while state dependence is handled by the adjoint. This is still reduced/implicit differentiation, but it should be disclosed rather than described as fully analytic differentiation.
4. Symmetry grouping is used in optimization to reduce 177 physical member sizes to 104 independent symmetric size groups; every member retains a size, with mirror-equivalent members tied by the symmetric geometry/loading assumption.
5. Optimization performance, not FE correctness, is now the bottleneck. The first wrapper accidentally requested all gradients during line-search value evaluations; this was identified and corrected before accepting any optimization result.
