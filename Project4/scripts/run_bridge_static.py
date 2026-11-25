import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, solve_static
from src.bridge import build_footbridge
from src.loads import assemble_bridge_loads, add_lumped_deck_mass, deck_plate_strip_check
from src.recovery import recover_element_forces_and_stress, member_stress_summary, total_member_mass, equilibrium_residual


def main():
    model = build_footbridge(H=2.0, camber=0.0)
    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, element_loads, load_meta = assemble_bridge_loads(model, cache)
    M_with_deck, deck_mass_lumped = add_lumped_deck_mass(M, model, cache)
    u, reactions, free = solve_static(K, F, model.fixed_dofs)

    elem = recover_element_forces_and_stress(model, u, cache, element_loads)
    mem = member_stress_summary(model, elem)
    steel_mass = total_member_mass(model)
    exact_deck_mass = 7850.0 * 0.008 * model.parameters["span"] * model.parameters["width"]
    force_resid, moment_resid = equilibrium_residual(model.nodes, F, reactions)

    deck_node_ids = [i for i, xyz in enumerate(model.nodes) if abs(xyz[2]) < 1e-10]
    uz = np.array([u[6*i+2] for i in deck_node_ids])
    max_down = float(max(0.0, -uz.min()))
    max_vm = float(max(r["max_von_mises"] for r in elem))
    plate_sigma = deck_plate_strip_check(
        model.parameters["deck_support_spacing"], load_meta["total_deck_pressure"]
    )

    out = {
        "milestone": 2,
        "model": {
            "nodes": len(model.nodes),
            "physical_members": len(model.members),
            "finite_elements": len(model.elements),
            "dofs": 6 * len(model.nodes),
            "free_dofs": len(free),
            "span_m": model.parameters["span"],
            "width_m": model.parameters["width"],
            "truss_depth_m": model.parameters["H"],
            "camber_m": model.parameters["camber"],
            "initial_radius_m": 0.050,
            "deck_support_spacing_m": model.parameters["deck_support_spacing"],
        },
        "loads": load_meta,
        "mass": {
            "steel_members_kg": steel_mass,
            "deck_exact_kg": exact_deck_mass,
            "deck_lumped_into_mass_matrix_kg": deck_mass_lumped,
            "total_kg": steel_mass + exact_deck_mass,
        },
        "static": {
            "max_deck_downward_deflection_m": max_down,
            "max_von_mises_pa": max_vm,
            "fatigue_limit_pa": 53.3e6,
            "static_strength_limit_pa": 160e6,
            "force_balance_residual_N": force_resid.tolist(),
            "moment_balance_residual_Nm": moment_resid.tolist(),
        },
        "deck_D2_hand_check": {
            "plate_strip_stress_pa": float(plate_sigma),
            "fatigue_limit_pa": 53.3e6,
            "passes": bool(plate_sigma <= 53.3e6),
        },
        "top_stressed_members": sorted(mem, key=lambda x: x["max_von_mises"], reverse=True)[:12],
    }

    path = ROOT / "assets" / "milestone2_results.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"nodes={len(model.nodes)} members={len(model.members)} elements={len(model.elements)} dofs={6*len(model.nodes)}")
    print(f"steel mass={steel_mass:.1f} kg, deck mass={exact_deck_mass:.1f} kg, total={steel_mass+exact_deck_mass:.1f} kg")
    print(f"max deck deflection={1e3*max_down:.3f} mm")
    print(f"max von Mises={max_vm/1e6:.3f} MPa")
    print(f"D2 plate strip stress={plate_sigma/1e6:.3f} MPa -> {'PASS' if plate_sigma<=53.3e6 else 'FAIL'}")
    print(f"force residual norm={np.linalg.norm(force_resid):.3e} N")
    print(f"moment residual norm={np.linalg.norm(moment_resid):.3e} N*m")
    print(f"deck lumped mass error={deck_mass_lumped-exact_deck_mass:.6e} kg")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
