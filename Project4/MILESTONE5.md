# Milestone 5 — Constrained bridge optimization (validated)

## Scope

The verified 3D bridge model is optimized using 54 symmetry-tied member-size variables plus two shape variables (`H` and top-chord camber), for **56 independent design variables** total. The FE model still contains all 177 physical members.

The objective is total mass (member steel + fixed deck mass). The governing requirements are:

- R2 fatigue/stress, enforced with the conservative stress KS aggregate;
- R3 deck-deflection KS;
- R4 lowest-frequency KS over the lowest six modes;
- R5 S1 linear-buckling KS over the six most critical buckling modes;
- R6 member slenderness, enforced conservatively with a KS aggregate and checked against the true maximum.

R1 static strength is reported but cannot govern once the tighter R2 fatigue requirement is satisfied.

## Numerical strategy

The optimization uses the verified reduced gradients from Milestone 4. The practical sequence is:

1. generate a feasible fully-stressed warm start;
2. use **NLopt LD_MMA** on the governing static constraints R2/R3/R6;
3. retain only independently reanalyzed feasible incumbents;
4. use short feasible sequential-linear-programming (SLP) refinements near the active set;
5. restore and independently evaluate R4 and R5 on the final candidate;
6. rerun V1–V6 regression tests.

A direct full-physics SLSQP polish was also tested. It reduced mass only by walking outside the active fatigue/slenderness feasible region, so those iterates were rejected rather than reported as an optimum.

## Final validated design

| Quantity | Initial design | Uniform feasible baseline | Optimized |
|---|---:|---:|---:|
| Total mass | 6,323.17 kg | 11,843.77 kg | **6,473.42 kg** |
| Member steel mass | 3,497.17 kg | 9,017.77 kg | **3,647.42 kg** |
| Truss depth `H` | 2.000 m | 3.000 m | **2.4683 m** |
| Top-chord camber | 0 m | 0 m | **0.4308 m** |
| True max von Mises stress | 173.15 MPa | 51.08 MPa | **50.67 MPa** |
| Fatigue factor of safety | 0.616 | 2.089 | **2.105** |
| Max deck deflection | 22.40 mm | 6.30 mm | **9.83 mm** |
| Lowest natural frequency | 4.778 Hz | 6.250 Hz | **3.767 Hz** |
| Buckling load factor | 21.61 | 62.18 | **9.02** |
| Max member slenderness | about 100 | 66.0 | **198.25** |

The failing 50 mm starting bridge is lighter than the optimum because it violates fatigue badly. The meaningful mass comparison is with the feasible uniform baseline. Relative to that baseline, the optimized design uses **59.55% less member steel** and **45.34% less total mass**.

## Active constraints

At the final design:

- stress KS = **-5.35e-5** (active to numerical resolution);
- slenderness KS = **-8.97e-7** (active);
- deflection KS = **-0.702** (inactive);
- vibration KS = **-0.204** (inactive);
- buckling KS = **-0.644** (inactive).

Engineering interpretation: fatigue sizes the heavily loaded load path, while R6 prevents lightly loaded bracing/vertical members from becoming unrealistically slender. Deflection, vibration, and S1 linear buckling are not governing once R2 and R6 are enforced.

## With and without R5

The design obtained without R5 has `alpha_cr = 9.02`, far above the required 3.0. Therefore restoring R5 leaves the locally refined design unchanged. This is the same qualitative outcome as the instructor's worked example: the slenderness rule already prevents the instability that R5 is meant to guard against.

## KKT diagnostic

A least-squares active-set multiplier estimate using the fatigue and slenderness aggregate gradients gives approximately:

- fatigue multiplier: **3,482 kg per unit relaxation of its normalized constraint**;
- slenderness multiplier: **83 kg per unit relaxation of its normalized constraint**.

The current free-variable relative stationarity residual is about **0.248**. Feasibility is tight, no lower-bound sign violations were detected, and a subsequent three-iteration feasible-SLP chunk accepted no mass-reducing feasible step. The multiplier/stationarity result should nevertheless be described as an **approximate KKT diagnostic**, not machine-precision stationarity.

## Regression status

After the optimization changes, V1–V6 were rerun from source. Beam statics, rotation invariance, frequency convergence, Euler buckling convergence, bridge equilibrium/deck checks, eigenvalue residuals, and the full 179-variable Taylor gradient check all pass.

## Files

- `assets/milestone5_results.json` — complete initial/baseline/final metrics, design vector, accepted-feasible history, R5 comparison, and KKT diagnostic.
- `assets/optimized_design.json` — final group/member radii and shape variables.
- `assets/final_full_check.json` — independent full-physics evaluation of the final design.
- `assets/optimization_history.png` — accepted-feasible mass history.
- `assets/active_constraint_margin.png` — convergence of the governing constraint margins.
