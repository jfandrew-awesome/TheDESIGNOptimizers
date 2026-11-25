import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, solve_static
from src.bridge import build_footbridge
from src.loads import assemble_bridge_loads, add_lumped_deck_mass, deck_plate_strip_check
from src.recovery import equilibrium_residual, recover_element_forces_and_stress


def main():
    model = build_footbridge(H=2.0, camber=0.0)
    assert len(model.members) == 177, len(model.members)
    assert len(model.elements) == 354, len(model.elements)
    assert np.all(model.element_member >= 0)
    assert len(model.fixed_dofs) > 0
    assert all(np.linalg.norm(model.nodes[j]-model.nodes[i]) > 1e-10 for i,j in model.elements)

    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, eload, meta = assemble_bridge_loads(model, cache)
    Md, added = add_lumped_deck_mass(M, model, cache)
    exact_deck_mass = 7850.0 * 0.008 * 15.0 * 3.0
    assert abs(added - exact_deck_mass) / exact_deck_mass < 1e-12

    plate = deck_plate_strip_check(model.parameters["deck_support_spacing"], meta["total_deck_pressure"])
    assert plate < 53.3e6

    u, R, free = solve_static(K, F, model.fixed_dofs)
    assert np.all(np.isfinite(u))
    assert np.linalg.norm(u[model.fixed_dofs]) == 0.0
    fr, mr = equilibrium_residual(model.nodes, F, R)
    force_scale = max(np.linalg.norm(F.reshape(-1,6)[:,:3].sum(axis=0)), 1.0)
    moment_scale = max(15.0 * force_scale, 1.0)
    assert np.linalg.norm(fr) / force_scale < 1e-10, (fr, force_scale)
    assert np.linalg.norm(mr) / moment_scale < 1e-10, (mr, moment_scale)

    stress = recover_element_forces_and_stress(model, u, cache, eload)
    assert len(stress) == len(model.elements)
    assert all(np.isfinite(r["max_von_mises"]) and r["max_von_mises"] >= 0 for r in stress)

    # Stiffness symmetry is a useful global assembly sanity check.
    asym = K - K.T
    assert np.linalg.norm(asym.data) < 1e-8 * max(np.linalg.norm(K.data), 1.0)

    deck_nodes = [i for i,x in enumerate(model.nodes) if abs(x[2]) < 1e-10]
    max_down = max(0.0, -min(u[6*i+2] for i in deck_nodes))
    max_vm = max(r["max_von_mises"] for r in stress)

    print("Bridge geometry/load/static checks: PASS")
    print(f"  nodes={len(model.nodes)}, physical members={len(model.members)}, FE elements={len(model.elements)}")
    print(f"  deck support spacing={model.parameters['deck_support_spacing']:.3f} m")
    print(f"  D2 plate stress={plate/1e6:.3f} MPa (<53.3 MPa)")
    print(f"  exact/lumped deck mass={exact_deck_mass:.3f}/{added:.3f} kg")
    print(f"  equilibrium |dF|={np.linalg.norm(fr):.3e} N, |dM|={np.linalg.norm(mr):.3e} N*m")
    print(f"  initial max deck deflection={max_down*1e3:.3f} mm")
    print(f"  initial max von Mises={max_vm/1e6:.3f} MPa")


if __name__ == "__main__":
    main()
