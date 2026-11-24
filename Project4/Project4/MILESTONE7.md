# Milestone 7 — Final report and reproducible submission package

## Completed

The GitHub Pages report in `index.md` now contains all required report sections:

1. summary and solver-generated headline results;
2. structure, variables, objective, explicit R1–R6 formulas/limits and sources;
3. FE/PDE analysis model, loads, supports, D2 deck treatment, and assumptions;
4. verification ladder with computed/reference/error data;
5. adjoint and eigenvalue-sensitivity derivations, V6 Taylor evidence, and timing comparison;
6. optimization strategy, accepted-feasible history, active constraints, and approximate KKT interpretation;
7. N1/S1 buckling component and with/without-R5 comparison;
8. initial/baseline/optimized results;
9. reproducibility instructions;
10. limitations and AI-use statement.

## Reproducibility improvement

`scripts/export_site_results.py` is now the single report-data build command. It:

- reanalyzes the initial and optimized bridges;
- rebuilds `assets/results.json`;
- recomputes V1–V6 report evidence;
- recomputes the current adjoint/finite-difference timing comparison;
- regenerates `assets/gradient_taylor.png`;
- regenerates `assets/optimization_history.png`;
- regenerates `assets/active_constraint_margin.png`.

All numerical result tables rendered by JavaScript read from `results.json` rather than a separately maintained copy.

## Final engineering result

The independent final analysis remains feasible for R1–R6. Fatigue/stress and slenderness are active; deflection, vibration, and buckling are inactive. The S1 buckling requirement is therefore unchanged by restoring R5 after the static sizing stage.

## Final validation performed

- `scripts/export_site_results.py`: PASS
- Python compilation of source, verification, and report-export modules: PASS
- `assets/results.json` parse: PASS
- `assets/report.js` ES-module syntax: PASS
- `assets/viewer.js` ES-module syntax: PASS
- all report asset references exist: PASS

Jekyll itself is not installed in the current execution environment, so the remaining validation is the live GitHub Pages/browser render.
