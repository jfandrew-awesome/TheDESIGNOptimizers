from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, solve_static
from src.bridge import build_footbridge
from src.eigen import natural_frequencies, assemble_geometric_stiffness, linear_buckling
from src.loads import assemble_bridge_loads, add_lumped_deck_mass, deck_plate_strip_check
from src.recovery import recover_element_forces_and_stress, member_stress_summary, total_member_mass
from src.optimization import slenderness_values
from src.sensitivity import analyze_design, response_values
from scripts.report_evidence import build_report_evidence

FATIGUE_LIMIT = 53.3e6
STATIC_LIMIT = 160e6
DEFLECTION_LIMIT = 15.0 / 400.0
FREQ_LIMIT = 3.0
BUCKLING_LIMIT = 3.0
SLENDER_LIMIT = 200.0
DECK_MASS = 7850.0 * 0.008 * 15.0 * 3.0


def _node_dofs(v):
    return np.asarray(v, float).reshape((-1, 6)).tolist()


def fatigue_fos(stress):
    return float(1.0 / ((stress / 2.0) / (0.7 * 86.2e6) + (stress / 2.0) / 460e6))


def analyze_for_site(name: str, radii, H, camber):
    model = build_footbridge(radii=np.asarray(radii, float), H=float(H), camber=float(camber))
    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, element_loads, load_meta = assemble_bridge_loads(model, cache)
    u, reactions, free = solve_static(K, F, model.fixed_dofs)
    Mtot, deck_mass = add_lumped_deck_mass(M, model, cache)
    vib = natural_frequencies(K, Mtot, model.fixed_dofs, n_modes=8)
    KG, axial = assemble_geometric_stiffness(model, u, cache, element_loads)
    buck = linear_buckling(K, KG, model.fixed_dofs, n_modes=8)
    elem = recover_element_forces_and_stress(model, u, cache, element_loads)
    mem = member_stress_summary(model, elem)

    max_vm = float(max(x['max_von_mises'] for x in elem))
    deck_nodes = [i for i, xyz in enumerate(model.nodes) if abs(xyz[2]) < 1e-10]
    max_down = float(max(0.0, -min(u[6*i+2] for i in deck_nodes)))
    lengths = np.array([m['length'] for m in model.members])
    radii_a = np.array([m['radius'] for m in model.members])
    slender = lengths / (0.673 * radii_a)
    member_mass = total_member_mass(model)
    plate_sigma = float(deck_plate_strip_check(model.parameters['deck_support_spacing'], load_meta['total_deck_pressure']))

    element_records = []
    for e, ((n1, n2), item, er) in enumerate(zip(model.elements, cache, elem)):
        idx, T, L, kl = item
        # Rows of R map global vectors to local vectors.
        R = T[:3, :3]
        element_records.append({
            'id': int(e),
            'nodes': [int(n1), int(n2)],
            'member': int(model.element_member[e]),
            'basis_global_to_local': R.tolist(),
            'length_m': float(L),
            'stress_pa': float(er['max_von_mises']),
            'stress_utilization_fatigue': float(er['max_von_mises'] / FATIGUE_LIMIT),
            'axial_force_N': float(axial[e]),
        })

    member_records = []
    for mid, (m, ms) in enumerate(zip(model.members, mem)):
        member_records.append({
            'id': mid,
            'name': m['name'],
            'kind': m['kind'],
            'radius_m': float(m['radius']),
            'length_m': float(m['length']),
            'stress_pa': float(ms['max_von_mises']),
            'stress_utilization_fatigue': float(ms['max_von_mises'] / FATIGUE_LIMIT),
            'slenderness': float(slender[mid]),
        })

    supports = []
    for nid, xyz in enumerate(model.nodes):
        dofs = set(range(6*nid, 6*nid+3))
        fixed_here = sorted(dofs.intersection(set(model.fixed_dofs.tolist())))
        if fixed_here:
            supports.append({'node': nid, 'position': np.asarray(xyz, float).tolist(), 'fixed_translations': [d-6*nid for d in fixed_here]})

    # A compact visual representation of the distributed deck load.  These are display arrows,
    # not an alternate discretization of the FE load vector.
    load_arrows = []
    for x in np.linspace(0.75, 14.25, 10):
        for y in (-1.125, -0.375, 0.375, 1.125):
            load_arrows.append({'position': [float(x), float(y), 0.35], 'direction': [0.0, 0.0, -1.0]})

    return {
        'name': name,
        'parameters': {
            'span_m': float(model.parameters['span']),
            'width_m': float(model.parameters['width']),
            'H_m': float(H),
            'camber_m': float(camber),
            'deck_thickness_m': 0.008,
            'deck_support_spacing_m': float(model.parameters['deck_support_spacing']),
        },
        'nodes_m': np.asarray(model.nodes, float).tolist(),
        'elements': element_records,
        'members': member_records,
        'static_dofs': _node_dofs(u),
        'vibration_mode_dofs': _node_dofs(vib['modes'][:, 0]),
        'buckling_mode_dofs': _node_dofs(buck['modes'][:, 0]),
        'supports': supports,
        'load_arrows': load_arrows,
        'metrics': {
            'mass_total_kg': float(member_mass + DECK_MASS),
            'mass_members_kg': float(member_mass),
            'deck_mass_kg': float(deck_mass),
            'max_stress_mpa': max_vm / 1e6,
            'static_strength_limit_mpa': STATIC_LIMIT / 1e6,
            'fatigue_limit_mpa': FATIGUE_LIMIT / 1e6,
            'fatigue_factor_of_safety': fatigue_fos(max_vm),
            'max_deflection_mm': max_down * 1000.0,
            'deflection_limit_mm': DEFLECTION_LIMIT * 1000.0,
            'lowest_frequency_hz': float(vib['frequencies_hz'][0]),
            'frequency_limit_hz': FREQ_LIMIT,
            'buckling_factor': float(buck['load_factors'][0]),
            'buckling_limit': BUCKLING_LIMIT,
            'max_slenderness': float(np.max(slender)),
            'slenderness_limit': SLENDER_LIMIT,
            'deck_plate_stress_mpa': plate_sigma / 1e6,
        },
    }


def main():
    m5 = json.loads((ROOT / 'assets' / 'milestone5_results.json').read_text())
    opt = m5['optimized_design']
    initial = analyze_for_site('Initial design', np.full(177, 0.05), 2.0, 0.0)
    optimized = analyze_for_site('Optimized design', opt['member_radii_m'], opt['H_m'], opt['camber_m'])

    history = []
    for i, h in enumerate(m5.get('history', [])):
        history.append({
            'step': i,
            'stage': h['stage'],
            'mass_total_kg': float(h['mass_total_kg']),
            'stress_ks': float(h.get('stress_ks', 0.0)),
            'slenderness_ks': float(h.get('slenderness_ks', 0.0)),
        })

    verification = build_report_evidence(history)

    result = {
        'schema_version': 2,
        'project': 'Project 4: PDE-Constrained Design Optimization',
        'structure': '15 m steel Pratt-style footbridge',
        'model': {
            'physical_members': 177,
            'finite_elements': 354,
            'deck_model': 'D2: loads and mass, no deck stiffness',
            'nonlinear_component': 'N1/S1 linear buckling about the computed static state',
            'design_variables': '54 symmetric member-size groups + H + camber',
        },
        'limits': {
            'R1_static_strength_mpa': 160.0,
            'R2_fatigue_fos_min': 2.0,
            'R2_equivalent_stress_limit_mpa': 53.3,
            'R3_deflection_mm_max': 37.5,
            'R4_frequency_hz_min': 3.0,
            'R5_buckling_factor_min': 3.0,
            'R6_slenderness_max': 200.0,
        },
        'initial': initial,
        'optimized': optimized,
        'baseline': {
            'mass_total_kg': float(m5['uniform_feasible_baseline']['mass_total_kg']),
            'mass_members_kg': float(m5['uniform_feasible_baseline']['mass_members_kg']),
            'H_m': float(m5['uniform_feasible_baseline']['H_m']),
            'camber_m': float(m5['uniform_feasible_baseline']['camber_m']),
            'max_stress_mpa': float(m5['uniform_feasible_baseline']['true_max_stress_pa']) / 1e6,
            'fatigue_factor_of_safety': float(m5['uniform_feasible_baseline']['fatigue_factor_of_safety']),
            'max_deflection_mm': float(m5['uniform_feasible_baseline']['true_max_down_m']) * 1e3,
            'lowest_frequency_hz': float(m5['uniform_feasible_baseline']['lowest_frequency_hz']),
            'buckling_factor': float(m5['uniform_feasible_baseline']['buckling_factor']),
            'max_slenderness': float(m5['uniform_feasible_baseline']['max_slenderness']),
        },
        'optimization_history': history,
        'verification': verification,
        'comparisons': m5.get('comparisons', {}),
        'optimization_constraints': {
            'stress_ks': float(opt['stress_ks']),
            'deflection_ks': float(opt['deflection_ks']),
            'vibration_ks': float(opt['vibration_ks']),
            'buckling_ks': float(opt['buckling_ks']),
            'slenderness_ks': float(opt['slenderness_ks']),
        },
        'kkt_diagnostic': m5.get('kkt_diagnostic', {}),
        'with_without_R5': {
            'optimization_without_R5_buckling_factor': float(optimized['metrics']['buckling_factor']),
            'R5_limit': 3.0,
            'accepted_incumbent_mass_kg': float(optimized['metrics']['mass_total_kg']),
            'buckling_ks': float(opt['buckling_ks']),
            'active_constraints': ['R2 fatigue/stress KS', 'R6 slenderness KS'],
            'full_physics_polish_outcome': 'A direct full-physics SLSQP polish attempted lower-mass steps, but those trial points lost feasibility through R2/R6 while R5 remained far inactive; rejected trial points were not reported as an optimum.',
            'conclusion': 'R5 is inactive at the locally refined design; restoring the full R2-R6 physics does not change the accepted incumbent because R2 and R6 already govern.'
        },
        'viewer_notes': {
            'member_thickness': 'Rendered at true CHS outer radius by default; the thickness control can exaggerate it for visibility.',
            'deformation': 'Static and eigenmode curves use cubic Euler-Bernoulli/Hermite interpolation with exported end translations and rotations.',
            'stress_color': 'Single-hue sequential scale of fatigue stress utilization. A geometric ring marks elements above the fatigue limit so failure is not conveyed by color alone.'
        }
    }

    out = ROOT / 'assets' / 'results.json'
    out.write_text(json.dumps(result, indent=2))
    print(f'wrote {out} ({out.stat().st_size/1024:.1f} KiB)')
    print('initial:', initial['metrics'])
    print('optimized:', optimized['metrics'])


if __name__ == '__main__':
    main()
