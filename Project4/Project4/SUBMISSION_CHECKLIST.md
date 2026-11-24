# Submission checklist

## Before pushing

- Optionally add team member names/course information near the top of `index.md`.
- Run `pip install -r requirements.txt` in the project environment if dependencies are not already installed.
- Run `python scripts/export_site_results.py` once from the repository root.
- Confirm the command completes and rewrites `assets/results.json` and the three report plots.

## GitHub

1. Create/use the team's **public** repository.
2. Copy the contents of this package to the repository root and push to `main`.
3. In **Settings → Pages**, select **Deploy from a branch**, branch **main**, folder **/(root)**.
4. Wait for the Pages deployment to finish and open the published site.

## Live-site check

Open the site in a private/incognito window and verify:

- MathJax equations render rather than showing raw TeX.
- Initial and optimized bridges appear side by side.
- Orbit/zoom affects both views together.
- The default member thickness is true radius; the thickness slider changes the display.
- Static deformation slider works.
- Lowest-vibration and buckling animations both work.
- Hovering a member shows its name/type, radius, stress, fatigue utilization, and slenderness.
- Metrics panels show pass/fail for R1–R6 and the D2 plate check.
- Taylor, optimization-history, and active-constraint plots load.
- The page remains usable at phone width.
- The report result table shows the optimized design at about 6.47 Mg total mass, fatigue FOS above 2, lowest frequency above 3 Hz, buckling factor above 3, and slenderness below 200.

## Submit

Submit both required links on Canvas:

- public GitHub repository URL;
- GitHub Pages site URL.
