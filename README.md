# Optimal Actuator Load Sharing

MAE 598 Design Optimization, Project 2: Ill-Conditioned Optimization.

Team repository: [TheDESIGNOptimizers](https://github.com/jfandrew-awesome/TheDESIGNOptimizers). Place these project files in a `project2/` subfolder to preserve the repository's existing root README.

**Main report:** [`actuator_load_sharing.ipynb`](actuator_load_sharing.ipynb). It contains all six required sections, the complete runnable code, and saved tables and figures. GitHub can display the notebook without running Python.

Two identical actuators share a 1000 N demand while minimizing a quadratic effort cost. The quadratic force-balance penalty produces eigenvalues 1 and 1 + 2 rho. Its condition number grows with rho and is unchanged by Jacobi scaling. An augmented Lagrangian enforces balance using a moderate penalty.

At a common dimensionless KKT tolerance of 1e-4:

| Method | Primal gradient updates | Gradient evaluations | Inner condition number |
|---|---:|---:|---:|
| Penalty gradient descent, rho = 10,000 | 168,122 | 168,123 | 20,001 |
| Augmented Lagrangian + gradient descent, beta = 1 | 246 | 255 | 3 |

The augmented Lagrangian uses nine multiplier updates. The approximately 683-fold reduction refers to primal update counts in this stated experiment, including every inner update. It is not a universal wall-clock speedup. Both final total-force errors are below 0.1 N. The exact constrained optimum is 500 N per actuator.

## Run the report

Use Python 3.12. From this directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_notebook.py
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_notebook.py
```

The runner executes every notebook cell in a fresh kernel using that environment and saves the outputs back into the notebook. It keeps kernel registration and configuration local to `.jupyter/`. No global kernel installation is needed. Alternatively, open the notebook in a Jupyter-compatible editor, select the environment containing the requirements, and run all cells in order. Numerical results use only NumPy and Matplotlib; the other packages support notebook execution.

The notebook needs no downloaded data and generally runs in under a minute on a typical desktop, although runtime varies. Seed 598 is recorded; the primary experiments are deterministic. It writes six figures, four CSV tables, and `summary.json` to `results/`. Small floating-point differences may occur across platforms.

`build_notebook.py` is an optional authoring utility. Running it regenerates the notebook source and clears outputs; run `run_notebook.py` afterward. Ordinary users only need to execute the existing notebook.

## Requirement coverage

| Requirement | Location |
|---|---|
| Real-world motivation | Notebook Section 1 |
| Variables, units, bounds, explicit objective/constraints, classification | Section 2 |
| Family G mechanism and hand-derived eigenvalues | Section 3 |
| D2: growth with rho and survival under Jacobi scaling | Section 3; `results/d2_intrinsic_test.png` |
| D1: logarithmic spectrum and condition numbers | Section 4; `results/d1_spectrum.png` |
| D3: baseline convergence, iteration counts, two-variable path | Section 4; `results/d3_*.png` |
| D4: remedy, common-accuracy comparison, conditioning change | Section 5; `results/d4_*.png` |
| Hand checks and executable verification | Sections 3-5 |
| Assumptions and simplifications | Section 6 |
| Reproducibility and AI-assistance disclosure | Setup, verification cell, and source notes |

## Before submission

1. Add the team member names to the notebook's opening cell.
2. Review the derivations, plots, and modeling assumptions so each team member can explain them.
3. Upload the `project2/` folder from `project2_actuator_load_sharing.zip` to the team's **public GitHub repository** linked above. It contains the notebook, this README, requirements, runner, optional builder, and `results/`. The archive excludes `.venv/` and `.jupyter/`.
4. Open the notebook on GitHub and confirm its equations, tables, and embedded plots render correctly. Local notebook validation and plot inspection have been completed; GitHub rendering cannot be checked until it is published.
5. Submit the public repository link on Canvas.

The project has been developed and executed locally. The existing public repository was inspected and contained only its root README. Upload and Canvas submission remain to be completed; the available browser session was signed out of GitHub.
