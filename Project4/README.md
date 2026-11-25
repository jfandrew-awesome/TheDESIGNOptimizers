# Project 4 — PDE-Constrained Footbridge Optimization

This repository contains the complete Project 4 submission: a from-scratch sparse 3D frame finite-element solver, verification ladder, adjoint/implicit sensitivities, state-dependent linear-buckling analysis, constrained sizing/shape optimization, and a GitHub Pages Three.js report for a 15 m steel footbridge.

## Build the report data

```bash
pip install -r requirements.txt
python scripts/export_site_results.py
```

The export command reanalyzes the designs, reruns the report verification evidence, regenerates the report plots, and writes the authoritative `assets/results.json` consumed by the report and 3D viewer.

## Verification

```bash
python verification/run_basic.py
python verification/run_linear_ladder.py
python verification/run_load_recovery.py
python verification/run_bridge_checks.py
python verification/run_eigen_bridge_checks.py
python verification/run_gradients.py
```

The final checkpoint passes V1–V6, including a directional Taylor test through the full 179-variable ungrouped design.

## Report/site

The GitHub Pages entry point is `index.md`. Supporting files are:

- `_config.yml`
- `_layouts/default.html`
- `assets/results.json`
- `assets/report.js`
- `assets/viewer.js`
- `assets/site.css`

Enable **Settings → Pages → Deploy from a branch → `main` / root**.

See `SUBMISSION_CHECKLIST.md` for the final live-site checks and `MILESTONE7.md` for the completed-report summary.
