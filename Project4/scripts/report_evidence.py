from __future__ import annotations

import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verification.run_basic import v1_cantilever, v1_second_plane
from verification.run_linear_ladder import simply_supported_midspan, free_free_modes, balance_test
from verification.run_eigen_bridge_checks import beam_frequency, euler_column, bridge_checks
from src.assembly import assemble, solve_static
from src.bridge import build_footbridge
from src.loads import assemble_bridge_loads, add_lumped_deck_mass, deck_plate_strip_check
from src.recovery import equilibrium_residual
from src.sensitivity import (
    analyze_design, response_values, reduced_gradients,
    pack_scaled_design, unpack_scaled_design, design_scales,
)

KEYS = ['mass_total_kg', 'stress_ks', 'deflection_ks', 'vibration_ks', 'buckling_ks']
LABELS = {
    'mass_total_kg': 'Mass',
    'stress_ks': 'Stress KS',
    'deflection_ks': 'Deflection KS',
    'vibration_ks': 'Vibration KS',
    'buckling_ks': 'Buckling KS',
}


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _gradient_taylor():
    base = build_footbridge()
    radii = np.array([m['radius'] for m in base.members])
    nm = len(radii)
    x0 = pack_scaled_design(radii, 2.0, 0.0)

    t0 = time.perf_counter()
    a0 = analyze_design(radii, 2.0, 0.0)
    analysis_seconds = time.perf_counter() - t0
    vals0 = response_values(a0)

    t0 = time.perf_counter()
    grads = reduced_gradients(a0)
    gradient_seconds = time.perf_counter() - t0

    rng = np.random.default_rng(34204)
    v = rng.normal(size=x0.size)
    v[-2:] = [2.0, -1.5]
    v /= np.linalg.norm(v)
    scales = design_scales(nm)
    directional = {k: float((grads[k] * scales) @ v) for k in KEYS}

    eps = np.array([1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4])
    errors = {k: [] for k in KEYS}

    def eval_x(xx):
        r, H, c = unpack_scaled_design(xx, nm)
        return response_values(analyze_design(r, H, c))

    for e in eps:
        vp = eval_x(x0 + e*v)
        vm = eval_x(x0 - e*v)
        for k in KEYS:
            cd = (vp[k] - vm[k]) / (2*e)
            err = abs(cd - directional[k]) / max(1.0, abs(cd), abs(directional[k]))
            errors[k].append(float(err))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in KEYS:
        ax.loglog(eps, errors[k], marker='o', label=LABELS[k])
    ax.set_xlabel('Central-difference step $\\epsilon$')
    ax.set_ylabel('Relative directional-derivative error')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'assets' / 'gradient_taylor.png', dpi=170)
    plt.close(fig)

    return {
        'variables': int(x0.size),
        'analysis_seconds': float(analysis_seconds),
        'all_reduced_gradients_seconds': float(gradient_seconds),
        'central_difference_full_jacobian_estimate_seconds': float(2*x0.size*analysis_seconds),
        'speedup_vs_full_central_difference': float((2*x0.size*analysis_seconds) / max(gradient_seconds, 1e-12)),
        'eps': eps.tolist(),
        'best_relative_errors': {k: float(min(errors[k])) for k in KEYS},
        'errors': errors,
        'directional_derivatives': directional,
        'response_values': {k: float(vals0[k]) for k in KEYS},
    }


def _optimization_plots(history):
    if not history:
        return
    step = np.arange(len(history))
    mass = np.array([float(h['mass_total_kg']) for h in history])
    stress = np.array([float(h.get('stress_ks', np.nan)) for h in history])
    slender = np.array([float(h.get('slenderness_ks', np.nan)) for h in history])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(step, mass, marker='o')
    ax.set_xlabel('Accepted feasible design')
    ax.set_ylabel('Total mass (kg)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / 'assets' / 'optimization_history.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(step, stress, marker='o', label='Fatigue/stress KS')
    ax.plot(step, slender, marker='s', label='Slenderness KS')
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel('Accepted feasible design')
    ax.set_ylabel('Constraint value (feasible $\\leq 0$)')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'assets' / 'active_constraint_margin.png', dpi=170)
    plt.close(fig)


def build_report_evidence(history=None):
    v1 = {**v1_cantilever(), **v1_second_plane()}
    v1_rows = [
        {'quantity': name, 'computed': float(got), 'exact': float(exact), 'relative_error': float(_rel(got, exact))}
        for name, (got, exact) in v1.items()
    ]

    d, ex = simply_supported_midspan()
    # Same arbitrary rotation used in the verification script.
    a = np.array([1., 2., 3.]); a /= np.linalg.norm(a); th = .73
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3)*np.cos(th) + (1-np.cos(th))*np.outer(a, a) + np.sin(th)*A
    dr, er = simply_supported_midspan(R=R)

    v3 = []
    for n in [1, 2, 4, 8, 16]:
        got, exact = beam_frequency(n)
        v3.append({'elements': n, 'computed_hz': float(got), 'exact_hz': float(exact), 'relative_error': float(_rel(got, exact))})

    v4 = []
    for n in [1, 2, 4, 8]:
        got, exact = euler_column(n)
        v4.append({'elements': n, 'computed_N': float(got), 'exact_N': float(exact), 'relative_error': float(_rel(got, exact))})

    nz, eigs = free_free_modes()
    scalar_balance = balance_test()

    model = build_footbridge(H=2.0, camber=0.0)
    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, eload, meta = assemble_bridge_loads(model, cache)
    Mdeck, added = add_lumped_deck_mass(M, model, cache)
    u, reactions, _ = solve_static(K, F, model.fixed_dofs)
    fr, mr = equilibrium_residual(model.nodes, F, reactions)
    plate = deck_plate_strip_check(model.parameters['deck_support_spacing'], meta['total_deck_pressure'])
    exact_deck_mass = 7850.0 * 0.008 * 15.0 * 3.0

    vib, buck, axial, rv, rb = bridge_checks()
    grad = _gradient_taylor()
    _optimization_plots(history or [])

    return {
        'V1_cantilever': {
            'rows': v1_rows,
            'max_relative_error': float(max(r['relative_error'] for r in v1_rows)),
        },
        'V2_simply_supported_rotation': {
            'computed_midspan_m': float(d),
            'exact_midspan_m': float(ex),
            'relative_error': float(_rel(d, ex)),
            'rotated_computed_midspan_m': float(dr),
            'rotated_relative_error': float(_rel(dr, er)),
            'rotation_invariance_relative_error': float(_rel(dr, d)),
        },
        'V3_frequency_convergence': v3,
        'V4_euler_buckling_convergence': v4,
        'V5_global_checks': {
            'free_free_zero_modes': int(nz),
            'expected_zero_modes': 6,
            'first_eigenvalues': np.asarray(eigs, float).tolist(),
            'scalar_balance_residual_N': float(scalar_balance),
            'bridge_force_residual_N': float(np.linalg.norm(fr)),
            'bridge_moment_residual_Nm': float(np.linalg.norm(mr)),
            'exact_deck_mass_kg': float(exact_deck_mass),
            'lumped_deck_mass_kg': float(added),
            'deck_plate_stress_mpa': float(plate / 1e6),
        },
        'bridge_eigen_residuals': {
            'vibration_relative_residual': float(rv),
            'buckling_relative_residual': float(rb),
            'initial_frequencies_hz': np.asarray(vib['frequencies_hz'], float).tolist(),
            'initial_buckling_factors': np.asarray(buck['load_factors'], float).tolist(),
            'initial_axial_force_range_kN': [float(axial.min()/1e3), float(axial.max()/1e3)],
        },
        'V6_gradient_taylor': grad,
    }
