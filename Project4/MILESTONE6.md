# Milestone 6 — Solver-generated website data and interactive 3D comparison

## Scope

This milestone converts the validated Milestone 5 FE/optimization solution into the GitHub Pages deliverable layer. No hand-entered geometry or viewer metrics are used: `scripts/export_site_results.py` rebuilds the initial and optimized bridges and writes the authoritative `assets/results.json` consumed by the site.

## Exported data

For both the initial and optimized design, `results.json` contains:

- actual nodal coordinates and 354 FE element connectivities;
- physical-member names, kinds, lengths, and CHS outer radii;
- the local element basis used by the FE transformation;
- element stress, fatigue utilization, and axial force;
- the converged static nodal translations and rotations;
- the lowest vibration eigenmode and first buckling eigenmode;
- support locations and display metadata for the distributed deck load;
- objective and R1–R6 values, plus the D2 deck-strip check.

The optimized data regenerate the Milestone 5 independent full check: 6473.415 kg total mass, 50.670 MPa peak stress, fatigue FOS 2.1055, 9.826 mm deck deflection, 3.76694 Hz lowest frequency, buckling factor 9.0195, and maximum slenderness 198.253.

## Viewer requirement mapping

1. **Initial/optimized comparison:** both are rendered side by side in one WebGL canvas. A single camera and OrbitControls instance renders both viewports, so the camera is intrinsically linked.
2. **True geometry:** every FE segment is drawn as a tube using its actual optimized or initial CHS outer radius at the default thickness setting. A user control can exaggerate thickness and the page states when that happens.
3. **Constraint field:** tubes are colored by fatigue-stress utilization using a single-hue sequential scale. Elements above the fatigue limit additionally receive a black geometric marker, so failure is not communicated by color alone.
4. **State on geometry:** the static deformed member centerline is evaluated with cubic Euler–Bernoulli/Hermite interpolation using both end translations and rotations exported from the FE solution. The deformation exaggeration is user-controlled.
5. **Animation:** the user can animate either the true lowest vibration eigenmode or the first buckling eigenmode. Eigenvectors are normalized only for display.
6. **Inspection:** hover ray-casting identifies an element/member and reports member type, radius, stress, fatigue utilization, and slenderness.
7. **Metrics:** solver-generated pass/fail panels report the objective and every R1–R6 value for both designs, plus the D2 deck plate check.
8. **Data/responsiveness:** the viewer fetches `assets/results.json`; CSS switches the metrics to one column on narrow screens. The current JSON is about 0.86 MiB.

## Reproducibility

Regenerate the viewer data with:

```bash
python scripts/export_site_results.py
```

The report shell also reads its headline/result table/model summary from `results.json`, so those displayed results are not separately maintained copies.

## Regression checks after site export

The FE source code was not modified for the viewer, but V1–V6 were rerun after packaging. The linear ladder, bridge static/global checks, bridge eigen checks, and full 179-variable V6 directional-gradient test all pass.

## Remaining work

The interactive deliverable shell exists, but the full written report still needs to be assembled around it: formulation, analysis assumptions, verification ladder/table, adjoint derivations, Taylor-test figure, optimization history/KKT discussion, nonlinear-component comparison, limitations, reproducibility instructions, and AI-use statement. That is the next milestone.
