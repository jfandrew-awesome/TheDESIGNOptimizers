from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def local_uniform_line_load(q_local, L):
    """Consistent 12-DOF nodal load for a uniform local line load [qx,qy,qz] (N/m)."""
    qx, qy, qz = np.asarray(q_local, dtype=float)
    f = np.zeros(12)
    # Axial load.
    f[[0, 6]] += qx * L / 2.0
    # Local +y transverse load, conjugate to v and rz.
    f[[1, 7]] += qy * L / 2.0
    f[5] += qy * L**2 / 12.0
    f[11] -= qy * L**2 / 12.0
    # Local +z transverse load. theta_y = -dw/dx for the adopted right-handed convention.
    f[[2, 8]] += qz * L / 2.0
    f[4] -= qz * L**2 / 12.0
    f[10] += qz * L**2 / 12.0
    return f


def assemble_bridge_loads(model, cache, pedestrian_pressure=5000.0, deck_thickness=0.008,
                          deck_density=7850.0, gravity=9.81):
    """Assemble member self-weight + D2 deck pressure/weight.

    Returns global F and a list of local equivalent nodal load vectors, one per FE element.
    """
    n_dof = 6 * len(model.nodes)
    F = np.zeros(n_dof)
    element_loads_local = [np.zeros(12) for _ in model.elements]
    q_deck_area = pedestrian_pressure + deck_density * gravity * deck_thickness
    gvec = np.array([0.0, 0.0, -gravity])

    for e, ((n1, n2), props, item) in enumerate(zip(model.elements, model.props_by_element, cache)):
        idx, T, L, _ = item
        R = T[:3, :3]
        # Uniform self-weight of this structural member.
        q_global = props["rho"] * props["A"] * gvec
        q_local = R @ q_global
        fl = local_uniform_line_load(q_local, L)

        # D2 deck load acts on the five longitudinal deck support lines only.
        if e in model.deck_support_element_width:
            trib = model.deck_support_element_width[e]
            q_global_deck = np.array([0.0, 0.0, -q_deck_area * trib])
            fl += local_uniform_line_load(R @ q_global_deck, L)

        fg = T.T @ fl
        F[idx] += fg
        element_loads_local[e] = fl

    return F, element_loads_local, {
        "pedestrian_pressure": pedestrian_pressure,
        "deck_self_weight_pressure": deck_density * gravity * deck_thickness,
        "total_deck_pressure": q_deck_area,
    }


def add_lumped_deck_mass(M, model, cache, deck_thickness=0.008, deck_density=7850.0):
    """D2 deck mass: lump tributary plate mass to the nodes of the support members."""
    Ml = lil_matrix(M)
    total_added = 0.0
    for e, item in enumerate(cache):
        if e not in model.deck_support_element_width:
            continue
        idx, _T, L, _kl = item
        trib = model.deck_support_element_width[e]
        m = deck_density * deck_thickness * trib * L
        total_added += m
        # Point mass m/2 at each element end; each point mass acts in x,y,z translations.
        for local_node in (0, 1):
            base = 6 * local_node
            for j in range(3):
                Ml[idx[base + j], idx[base + j]] += m / 2.0
    return csr_matrix(Ml), total_added


def deck_plate_strip_check(spacing, q_total, thickness=0.008):
    """D2 hand check: simply supported unit-width strip under uniform pressure."""
    sigma = 0.75 * q_total * spacing**2 / thickness**2
    return sigma
