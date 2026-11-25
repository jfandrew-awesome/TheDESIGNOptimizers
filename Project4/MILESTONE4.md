# Milestone 4 — V6 reduced sensitivities and Taylor verification

This milestone adds reduced gradients for the full bridge design vector:

- 177 physical-member radii;
- truss depth `H`;
- top-chord camber.

There are **179 design variables** total.

## Differentiation strategy

The bridge uses the reduced-space/adjoint formulation around the sparse static solve

\[
K(d)u=F(d).
\]

For a scalar response `g(d,u)`, the code solves

\[
K^T\lambda=-g_u^T,
\qquad
\frac{dg}{dd_j}=g_{d_j}+\lambda^T(K_{d_j}u-F_{d_j}).
\]

Member-radius derivatives of section properties, element stiffness, consistent mass, and
self-weight are analytical and element-local. The two geometry variables use centered
partial differentiation of the assembled operators **at fixed state**; their state dependence
still goes through the same adjoint equation, so no perturbed PDE solves are used inside the
reduced-gradient routine.

Implemented scalar responses:

1. total mass (including fixed deck mass);
2. KS-aggregated fatigue/stress utilization using the 53.3 MPa limit;
3. KS-aggregated downward deck deflection using the 37.5 mm limit;
4. KS aggregation of the lowest six vibration modes, using eigenvalue sensitivities;
5. KS aggregation of the six most critical buckling eigenvalues, including the adjoint term
   through the static axial forces that build `K_G`.

The buckling aggregate combines all six state terms before solving the adjoint, so it still
requires only **one static adjoint solve** for that aggregate.

## V6 directional Taylor test

The check uses a deterministic random direction in the **scaled 179-variable design space**
and deliberately gives nontrivial components to both `H` and camber. Adjoint/eigen directional
derivatives are compared with central differences of the complete analysis.

Best relative directional-derivative errors over the step sweep:

| Response | Best relative error |
|---|---:|
| Total mass | 1.37e-11 |
| Stress KS | 6.74e-08 |
| Deflection KS | 9.58e-10 |
| Vibration KS | 2.38e-10 |
| Buckling KS | 3.52e-09 |

The Taylor curves fall at approximately second order until numerical noise, as expected for a
central-difference verification.

The plot is written to `assets/gradient_taylor.png` and the machine-readable data to
`assets/milestone4_results.json`.

## Cost comparison

On the current runtime, one complete bridge analysis took about 0.58 s and all five reduced
gradients took about 7.21 s. A full 179-variable central-difference Jacobian would require
358 complete analyses, estimated at about 208 s on the same runtime. Thus the current reduced
gradient implementation is roughly **29x faster** than brute-force central differences even
before further caching/optimization.

## Run

```bash
python verification/run_gradients.py
python scripts/run_bridge_gradients.py
```

## Next milestone

Use these verified gradients in a scaled constrained optimizer, add the explicit member
slenderness constraints, save the optimization history, and compare the optimum with/without
R5 buckling as required by the project.
