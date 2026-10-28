# Minimum submission checklist

- [ ] Fill the actual team roster, contact dates, and discovery method in `README.md`.
- [ ] Confirm the course-sheet reservation and, if needed, acceptance of the written exchange as the interview.
- [ ] Send the email below with `researcher-memo.pdf` or `researcher-memo.txt` attached. Record the actual send date. Record brief review only after Jimmie confirms or corrects it; include a reply if received.
- [ ] Check the worked arithmetic in the report and run `python src/verify.py` after installing `requirements.txt`.
- [ ] Upload the contents of this folder to a public team GitHub repository, check the report and images, and submit its URL on Canvas. Follow Canvas for the presentation schedule.

The folder has 33 files. Keep the three `results/runs-seed-*.zip` files zipped. They retain all 42 runs and are read directly by the analysis scripts.

## Email to Jimmie

Reply on your existing thread. Attach the one-page memo. The brief is in the email body, so no separate brief PDF is necessary.

**Subject:** Project 3 follow-up: brief and recommendation

Hi Jimmie,

Thanks again for your help. Our short problem brief is below, and our one-page recommendation memo is attached. Could you let us know if the brief gets anything wrong?

Problem brief

Jimmie Adriazola's current research focuses on nonlinear waves, integrable and
near-integrable systems, model reduction, and scientific machine learning. He is
interested in which spectral structures survive when integrable systems are
perturbed and whether those structures provide useful coordinates for computation
and learning. PDE-constrained optimal control and his Maxwell work are earlier
research.

His main concern is fidelity over time: small short-time error can grow into
wrong phase, conserved quantities, or coherent structures. This matters when
interpreting nonlinear wave behavior in optical, fluid, and quantum settings.

His projects use different PDEs and a mix of spectral methods, numerical PDE
solvers, symbolic work, and learned models. He did not specify one production
equation, solver runtime, hardware setup, or query count. Standard PINNs are not
a primary tool in his work. He prefers enforcing known constraints directly
when practical and judges accuracy against the application's modeling and
discretization errors.

For this project, he suggested a periodic cubic nonlinear Schrodinger test with
a Fourier reference. We use a nonconstant initial condition and increasing time
horizons to examine phase, mass, Hamiltonian, and solution error. This is our
synthetic case study, not a failed experiment he reported.

Thanks,
Ashden
