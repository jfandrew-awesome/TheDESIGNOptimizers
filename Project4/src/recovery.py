from __future__ import annotations

import numpy as np


def recover_element_forces_and_stress(model, u, cache, element_loads_local):
    """Recover local element end forces and project-specified CHS von Mises stress."""
    results = []
    for e, ((n1, n2), props, item, fl) in enumerate(
        zip(model.elements, model.props_by_element, cache, element_loads_local)
    ):
        idx, T, L, kl = item
        ul = T @ u[idx]
        q = kl @ ul - fl
        end_data = []
        for off in (0, 6):
            N = q[off + 0]
            Tq = q[off + 3]
            My = q[off + 4]
            Mz = q[off + 5]
            sigma = abs(N) / props["A"] + np.hypot(My, Mz) * props["r"] / props["Iy"]
            tau = abs(Tq) * props["r"] / props["J"]
            vm = np.sqrt(sigma**2 + 3.0 * tau**2)
            end_data.append({
                "N": float(N), "T": float(Tq), "My": float(My), "Mz": float(Mz),
                "sigma": float(sigma), "tau": float(tau), "von_mises": float(vm),
            })
        results.append({
            "element": e,
            "member": int(model.element_member[e]),
            "nodes": [int(n1), int(n2)],
            "length": float(L),
            "ends": end_data,
            "max_von_mises": max(end_data[0]["von_mises"], end_data[1]["von_mises"]),
        })
    return results


def member_stress_summary(model, element_results):
    out = []
    for mid, member in enumerate(model.members):
        vals = [r["max_von_mises"] for r in element_results if r["member"] == mid]
        out.append({
            "member": mid,
            "name": member["name"],
            "kind": member["kind"],
            "radius": member["radius"],
            "length": member["length"],
            "max_von_mises": float(max(vals)),
        })
    return out


def total_member_mass(model):
    mass = 0.0
    for m in model.members:
        # All FE children of a physical member share the same section.
        # Pull its first child section through the member radius formula indirectly via A below.
        r = m["radius"]
        ri = 0.9 * r
        A = np.pi * (r**2 - ri**2)
        mass += model.parameters["rho"] * A * m["length"]
    return float(mass)


def equilibrium_residual(nodes, F, reactions):
    """Return net generalized force/moment after adding loads and support reactions."""
    total = np.asarray(F) + np.asarray(reactions)
    force = np.zeros(3)
    moment = np.zeros(3)
    for i, x in enumerate(nodes):
        fi = total[6*i:6*i+3]
        mi = total[6*i+3:6*i+6]
        force += fi
        moment += np.cross(x, fi) + mi
    return force, moment
