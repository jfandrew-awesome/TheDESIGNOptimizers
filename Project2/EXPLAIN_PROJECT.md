# How to explain this project

Read `REPORT.md` for the submission. Use this page to prepare what you will say.

## The 30-second version

“Our project has two actuators sharing a 1000-newton load. We want them to supply the required force with minimum effort. A large penalty can enforce force balance, but it makes gradient descent very slow. We demonstrate why that happens and use an augmented Lagrangian to fix it. At the same accuracy target, the number of gradient updates falls from 168,122 to 246.”

## A simple walk through the six figures

**Figure 1 — What is the problem?**

“Both actuators need to add up to 1000 newtons. We charge effort based on force squared, so an equal 500/500 split is better than 900/100. We already know the ideal answer; the project is about how the solver reaches it.”

**Figure 2 — Why is it difficult numerically?**

“There are two types of motion. We can move load from one actuator to the other without changing the total. Or we can change the total force. A large penalty strongly resists the second motion. The solver has to take small steps, even in the direction where bigger steps would be acceptable.”

**Figure 3 — Is that just a scaling mistake?**

“No. We increase the penalty and watch the condition number grow. Then we apply the required diagonal scaling. It does not improve the condition number. That passes both parts of the assignment's intrinsic-conditioning test.”

**Figures 4 and 5 — What does the slowdown look like?**

“The larger penalties need more updates. The contour plot shows gradient descent zig-zagging through a narrow valley. We use an analytically favorable constant step, so we are not making the baseline slow by choosing an arbitrary step.”

**Figure 6 — How do we fix it?**

“The augmented Lagrangian adds a correction based on the remaining imbalance. We repeatedly adjust the forces, measure the error, and update that correction. This lets us keep the penalty small. We count every inner gradient update, and both methods have to satisfy the same force-balance and optimality checks.”

## Terms in everyday language

| Term | What to say |
|---|---|
| Objective | The effort cost we want to minimize. |
| Constraint | The rule that the two forces must add to 1000 N. |
| Penalty weight, rho | How expensive we make breaking that rule. |
| Gradient descent | A method that repeatedly takes a downhill step. |
| Hessian | A matrix that tells us how curved the objective is in different directions. |
| Condition number, kappa | The ratio of the largest to smallest curvature in this positive-definite problem. |
| Jacobi scaling | Rescaling each coordinate separately using the Hessian's diagonal. |
| Multiplier, y | A signed correction that accumulates the force-balance error. |
| Inner iteration | One step while adjusting the forces for a fixed correction. |
| Outer iteration | One round of force adjustment followed by a correction update. |
| KKT residual | A combined check that force balance and optimality are both accurate. |

## Questions you may be asked

**Why not just use 500 N each?**

That is the practical answer for this simplified system. We deliberately chose a small problem whose exact answer is known so we can verify the mathematics and explain how the penalty changes the solver's behavior.

**Is the physical system itself ill-conditioned?**

The original constrained allocation problem is well behaved. The large condition number is introduced by the penalty formulation. That is family G in the assignment.

**Why does diagonal scaling fail?**

The two diagonal entries are equal. Scaling them divides both eigenvalues by the same factor, so their ratio stays the same.

**Why does the baseline table say 138,163, but the final comparison says 168,122?**

They use different stopping rules. The first studies each penalty objective using a relative gradient tolerance. The final comparison requires both methods to meet the same absolute force-balance and optimality tolerance for the original problem.

**Did the second method take only nine steps?**

No. It took nine outer correction updates containing 246 total gradient updates. We report all 246, not just the nine outer rounds.

**Is it 683 times faster?**

It uses about 683 times fewer gradient updates in this experiment. We do not claim a universal runtime speedup.

**Are the intermediate forces applied to hardware?**

No. They are offline numerical iterates. The model allows signed forces and temporary imbalance; a real controller would need additional physical limits and dynamics.

**Do I need to explain every line of code?**

Focus on the model, why the conditioning grows, what each graph demonstrates, and how the stopping rules make the comparison meaningful. Be ready to identify the update equations and checks in the code.
