const data = await (await fetch('assets/results.json')).json();

function fmt(v, digits=2) {
  return Number(v).toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
}
function sci(v, digits=2) { return Number(v).toExponential(digits); }
function setHTML(id, html) { const el=document.getElementById(id); if (el) el.innerHTML=html; }

const i=data.initial.metrics, o=data.optimized.metrics, ub=data.baseline;
setHTML('headline-result', `<strong>Headline result.</strong> The optimized bridge uses <strong>${fmt(o.mass_members_kg,0)} kg of member steel</strong>, ${fmt(data.comparisons.member_mass_reduction_vs_uniform_feasible_pct,1)}% less than the lightest uniform feasible baseline in our model, while satisfying R1–R6. Fatigue and member slenderness govern the final design.`);

const resultRows = [
  ['Total mass', `${fmt(i.mass_total_kg,0)} kg`, `${fmt(ub.mass_total_kg,0)} kg`, `${fmt(o.mass_total_kg,0)} kg`],
  ['Member steel mass', `${fmt(i.mass_members_kg,0)} kg`, `${fmt(ub.mass_members_kg,0)} kg`, `${fmt(o.mass_members_kg,0)} kg`],
  ['Truss depth H', `${fmt(data.initial.parameters.H_m,3)} m`, `${fmt(ub.H_m,3)} m`, `${fmt(data.optimized.parameters.H_m,3)} m`],
  ['Top-chord camber', `${fmt(data.initial.parameters.camber_m,3)} m`, `${fmt(ub.camber_m,3)} m`, `${fmt(data.optimized.parameters.camber_m,3)} m`],
  ['Max von Mises stress', `${fmt(i.max_stress_mpa,2)} MPa`, `${fmt(ub.max_stress_mpa,2)} MPa`, `${fmt(o.max_stress_mpa,2)} MPa`],
  ['Fatigue factor of safety', fmt(i.fatigue_factor_of_safety,3), fmt(ub.fatigue_factor_of_safety,3), fmt(o.fatigue_factor_of_safety,3)],
  ['Max deck deflection', `${fmt(i.max_deflection_mm,2)} mm`, `${fmt(ub.max_deflection_mm,2)} mm`, `${fmt(o.max_deflection_mm,2)} mm`],
  ['Lowest natural frequency', `${fmt(i.lowest_frequency_hz,3)} Hz`, `${fmt(ub.lowest_frequency_hz,3)} Hz`, `${fmt(o.lowest_frequency_hz,3)} Hz`],
  ['Buckling load factor', fmt(i.buckling_factor,2), fmt(ub.buckling_factor,2), fmt(o.buckling_factor,2)],
  ['Max member slenderness', fmt(i.max_slenderness,2), fmt(ub.max_slenderness,2), fmt(o.max_slenderness,2)],
];
setHTML('result-summary-table', `<table class="report"><thead><tr><th>Metric</th><th>Initial</th><th>Uniform feasible baseline</th><th>Optimized</th></tr></thead><tbody>${resultRows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
setHTML('result-interpretation', `<p>The 50 mm starting bridge is lighter than the optimized bridge only because it is infeasible: its fatigue factor of safety is ${fmt(i.fatigue_factor_of_safety,3)}. The meaningful mass comparison is therefore the feasible uniform baseline. Relative to that baseline, optimization removes <strong>${fmt(data.comparisons.member_mass_reduction_vs_uniform_feasible_pct,1)}%</strong> of member steel and <strong>${fmt(data.comparisons.total_mass_reduction_vs_uniform_feasible_pct,1)}%</strong> of total mass.</p>`);
setHTML('model-optimization-text', `<p>The production model contains <strong>${data.model.physical_members}</strong> physical CHS members and <strong>${data.model.finite_elements}</strong> 3D Euler–Bernoulli frame elements. Every physical member is divided into two FE elements. The deck uses option D2: pressure and mass are transferred to five longitudinal support lines while the plate stiffness is omitted and checked separately.</p><p>The optimization uses <strong>${data.model.design_variables}</strong>. Sizes are tied only across existing x/y reflection-equivalent members, while the full 177-member FE model is analyzed at every accepted design.</p>`);

const v=data.verification;
const v3=v.V3_frequency_convergence.at(-1);
const v4=v.V4_euler_buckling_convergence.at(-1);
const best=v.V6_gradient_taylor.best_relative_errors;
const worstGrad=Math.max(...Object.values(best));
const verRows = [
  ['V1 cantilever statics/torsion', '5 closed-form responses', 'closed-form beam/torsion formulas', sci(v.V1_cantilever.max_relative_error,2)],
  ['V2 simply supported beam', `${fmt(v.V2_simply_supported_rotation.computed_midspan_m*1e3,6)} mm`, `${fmt(v.V2_simply_supported_rotation.exact_midspan_m*1e3,6)} mm`, sci(v.V2_simply_supported_rotation.relative_error,2)],
  ['V2 3D rotation invariance', `${fmt(v.V2_simply_supported_rotation.rotated_computed_midspan_m*1e3,6)} mm`, 'unrotated response', sci(v.V2_simply_supported_rotation.rotation_invariance_relative_error,2)],
  ['V3 first beam frequency, 16 elements', `${fmt(v3.computed_hz,9)} Hz`, `${fmt(v3.exact_hz,9)} Hz`, sci(v3.relative_error,2)],
  ['V4 Euler column, 8 elements', `${fmt(v4.computed_N,3)} N`, `${fmt(v4.exact_N,3)} N`, sci(v4.relative_error,2)],
  ['V5 free-free rigid modes', `${v.V5_global_checks.free_free_zero_modes}`, `${v.V5_global_checks.expected_zero_modes}`, 'exact count'],
  ['V5 bridge force balance', `${sci(v.V5_global_checks.bridge_force_residual_N,2)} N`, '0 N', 'numerical residual'],
  ['V5 bridge moment balance', `${sci(v.V5_global_checks.bridge_moment_residual_Nm,2)} N·m`, '0 N·m', 'numerical residual'],
  ['Bridge vibration eigen residual', sci(v.bridge_eigen_residuals.vibration_relative_residual,2), '0', 'relative residual'],
  ['Bridge buckling eigen residual', sci(v.bridge_eigen_residuals.buckling_relative_residual,2), '0', 'relative residual'],
  ['V6 adjoint/Taylor check', '5 reduced responses', 'central difference', `worst best-error ${sci(worstGrad,2)}`],
];
setHTML('verification-summary-table', `<table class="report"><thead><tr><th>Test</th><th>Computed</th><th>Reference</th><th>Error/check</th></tr></thead><tbody>${verRows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table>`);

const cost=v.V6_gradient_taylor;
setHTML('sensitivity-cost-table', `<table class="report"><thead><tr><th>Quantity</th><th>Measured/estimated cost</th></tr></thead><tbody>
<tr><td>One complete bridge analysis</td><td>${fmt(cost.analysis_seconds,3)} s</td></tr>
<tr><td>All five reduced gradients</td><td>${fmt(cost.all_reduced_gradients_seconds,3)} s</td></tr>
<tr><td>179-variable central-difference Jacobian (358 analyses)</td><td>${fmt(cost.central_difference_full_jacobian_estimate_seconds,1)} s estimated</td></tr>
<tr><td>Reduced-gradient speedup</td><td>≈ ${fmt(cost.speedup_vs_full_central_difference,1)}×</td></tr>
</tbody></table>`);

const gradRows = Object.entries(best).map(([k,val])=>`<tr><td>${k.replaceAll('_',' ')}</td><td>${sci(val,3)}</td></tr>`).join('');
setHTML('gradient-error-table', `<table class="report"><thead><tr><th>Response</th><th>Best relative directional error</th></tr></thead><tbody>${gradRows}</tbody></table>`);


const oc=data.optimization_constraints;
setHTML('active-constraint-table', `<table class="report"><thead><tr><th>Aggregate constraint</th><th>Final value</th><th>Status</th></tr></thead><tbody>
<tr><td>R2 fatigue/stress KS</td><td>${sci(oc.stress_ks,3)}</td><td>Active</td></tr>
<tr><td>R3 deflection KS</td><td>${fmt(oc.deflection_ks,3)}</td><td>Inactive</td></tr>
<tr><td>R4 vibration KS</td><td>${fmt(oc.vibration_ks,3)}</td><td>Inactive</td></tr>
<tr><td>R5 buckling KS</td><td>${fmt(oc.buckling_ks,3)}</td><td>Inactive</td></tr>
<tr><td>R6 slenderness KS</td><td>${sci(oc.slenderness_ks,3)}</td><td>Active</td></tr>
</tbody></table>`);

const k=data.kkt_diagnostic;
setHTML('kkt-table', `<table class="report"><thead><tr><th>Diagnostic</th><th>Value</th></tr></thead><tbody>
<tr><td>Fatigue multiplier</td><td>${fmt(k.multiplier_estimate_stress_kg_per_unit_normalized_limit,1)} kg per unit normalized relaxation</td></tr>
<tr><td>Slenderness multiplier</td><td>${fmt(k.multiplier_estimate_slenderness_kg_per_unit_normalized_limit,1)} kg per unit normalized relaxation</td></tr>
<tr><td>Free-variable relative stationarity residual</td><td>${fmt(k.free_variable_relative_stationarity_residual,3)}</td></tr>
<tr><td>Lower-bound sign violations</td><td>${k.lower_bound_sign_violations}</td></tr>
</tbody></table>`);

setHTML('r5-comparison', `<p>Optimization was performed without R5 during the static sizing stage and R5 was then restored in the full-physics check. The resulting design has <strong>α<sub>cr</sub> = ${fmt(data.with_without_R5.optimization_without_R5_buckling_factor,2)}</strong> versus the required ${fmt(data.with_without_R5.R5_limit,1)}. Therefore the same locally refined design remains feasible with R5 included; R6 slenderness already prevents the critical instability from governing.</p>`);
