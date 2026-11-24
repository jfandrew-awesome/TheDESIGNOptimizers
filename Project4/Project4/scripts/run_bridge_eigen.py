import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, solve_static
from src.bridge import build_footbridge
from src.eigen import natural_frequencies, assemble_geometric_stiffness, linear_buckling
from src.loads import assemble_bridge_loads, add_lumped_deck_mass


def residual_norm(A, x, lam, B):
    r = A @ x - lam * (B @ x)
    den = np.linalg.norm(A @ x) + abs(lam) * np.linalg.norm(B @ x)
    return float(np.linalg.norm(r) / max(den, 1e-30))


def main():
    model = build_footbridge()
    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, element_loads_local, load_info = assemble_bridge_loads(model, cache)
    u, reactions, free = solve_static(K, F, model.fixed_dofs)

    Mtot, deck_mass = add_lumped_deck_mass(M, model, cache)
    vib = natural_frequencies(K, Mtot, model.fixed_dofs, n_modes=10)

    KG, axial = assemble_geometric_stiffness(model, u, cache, element_loads_local)
    buck = linear_buckling(K, KG, model.fixed_dofs, n_modes=10)

    # Residuals on supported matrices for the first eigenpair of each type.
    Kf = K[free][:, free]
    Mf = Mtot[free][:, free]
    phi = vib["modes"][free, 0]
    vib_res = residual_norm(Kf, phi, vib["eigenvalues"][0], Mf)

    psi = buck["modes"][free, 0]
    buck_res = residual_norm(-KG[free][:, free], psi, buck["mu"][0], Kf)

    compression = axial[axial < 0.0]
    tension = axial[axial > 0.0]
    out = {
        "initial_design": {"radius_m": 0.05, "H_m": 2.0, "camber_m": 0.0},
        "n_nodes": len(model.nodes),
        "n_elements": len(model.elements),
        "n_members": len(model.members),
        "lowest_frequency_hz": float(vib["frequencies_hz"][0]),
        "frequencies_hz": [float(x) for x in vib["frequencies_hz"]],
        "vibration_eigen_residual": vib_res,
        "buckling_load_factor": float(buck["load_factors"][0]),
        "buckling_load_factors": [float(x) for x in buck["load_factors"]],
        "buckling_eigen_residual": buck_res,
        "min_element_axial_force_N": float(np.min(axial)),
        "max_element_axial_force_N": float(np.max(axial)),
        "most_compressive_element_force_N": float(np.min(compression)) if len(compression) else None,
        "most_tensile_element_force_N": float(np.max(tension)) if len(tension) else None,
        "deck_mass_kg": float(deck_mass),
        "requirements": {
            "R4_frequency_limit_hz": 3.0,
            "R4_pass": bool(vib["frequencies_hz"][0] >= 3.0),
            "R5_buckling_factor_limit": 3.0,
            "R5_pass": bool(buck["load_factors"][0] >= 3.0),
        },
        "load_info": load_info,
    }
    path = ROOT / "assets" / "milestone3_results.json"
    path.write_text(json.dumps(out, indent=2))

    print("Bridge vibration and buckling")
    print(f"  lowest frequency       : {out['lowest_frequency_hz']:.6f} Hz")
    print(f"  first 6 frequencies    : {[round(x,6) for x in out['frequencies_hz'][:6]]}")
    print(f"  vibration eig residual : {vib_res:.3e}")
    print(f"  buckling load factor   : {out['buckling_load_factor']:.6f}")
    print(f"  first buckling factors : {[round(x,6) for x in out['buckling_load_factors'][:6]]}")
    print(f"  buckling eig residual  : {buck_res:.3e}")
    print(f"  axial force range      : {np.min(axial)/1e3:.3f} to {np.max(axial)/1e3:.3f} kN")
    print(f"  R4 pass                : {out['requirements']['R4_pass']}")
    print(f"  R5 pass                : {out['requirements']['R5_pass']}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
