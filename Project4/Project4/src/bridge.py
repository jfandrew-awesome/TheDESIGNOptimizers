from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .sections import chs_properties


@dataclass
class BridgeModel:
    nodes: np.ndarray
    elements: List[Tuple[int, int]]
    props_by_element: List[dict]
    element_member: np.ndarray
    members: List[dict]
    fixed_dofs: np.ndarray
    deck_support_element_width: Dict[int, float]
    parameters: dict


class _NodeRegistry:
    def __init__(self, tol_digits: int = 10):
        self.coords: List[np.ndarray] = []
        self._lookup: Dict[Tuple[float, float, float], int] = {}
        self._digits = tol_digits

    def get(self, xyz) -> int:
        xyz = np.asarray(xyz, dtype=float)
        key = tuple(np.round(xyz, self._digits))
        if key not in self._lookup:
            self._lookup[key] = len(self.coords)
            self.coords.append(xyz.copy())
        return self._lookup[key]


def build_footbridge(
    radii=None,
    H: float = 2.0,
    camber: float = 0.0,
    n_panels: int = 10,
    span: float = 15.0,
    width: float = 3.0,
    subdivisions: int = 2,
    E: float = 200e9,
    nu: float = 0.3,
    rho: float = 7850.0,
) -> BridgeModel:
    """Build the project's 3D footbridge concept.

    Physical concept:
      * two Pratt-style side trusses at y = +/- width/2,
      * floor beams at each panel line,
      * three interior stringers plus the two bottom chords as deck support lines,
      * top cross-beams and alternating top-plane lateral bracing,
      * each physical member split into >=2 Euler-Bernoulli beam elements.

    Shape variables are H (base truss depth) and camber (parabolic top-chord rise at midspan).
    """
    if subdivisions < 2:
        raise ValueError("project specification requires at least two beam elements per member")
    if n_panels % 2:
        raise ValueError("use an even number of panels for the symmetric Pratt pattern")
    if H <= 0 or width <= 0 or span <= 0:
        raise ValueError("positive geometry required")

    reg = _NodeRegistry()
    physical_members: List[dict] = []
    xj = np.linspace(0.0, span, n_panels + 1)
    ydeck = np.linspace(-width / 2.0, width / 2.0, 5)

    def top_z(x):
        xi = x / span
        return H + 4.0 * camber * xi * (1.0 - xi)

    def add_member(a, b, kind, name=None, deck_width=None):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if np.linalg.norm(b - a) < 1e-12:
            raise ValueError(f"zero-length physical member {kind}")
        physical_members.append({
            "a": a,
            "b": b,
            "kind": kind,
            "name": name or f"{kind}_{len(physical_members)}",
            "deck_width": deck_width,
        })

    # Side trusses. Bottom chords double as the two edge deck stringers.
    for side, y in enumerate((-width / 2.0, width / 2.0)):
        tag = "L" if side == 0 else "R"
        for i in range(n_panels):
            a = [xj[i], y, 0.0]; b = [xj[i + 1], y, 0.0]
            add_member(a, b, "bottom_chord", f"bottom_{tag}_{i}", deck_width=width / 8.0)
            add_member([xj[i], y, top_z(xj[i])], [xj[i + 1], y, top_z(xj[i + 1])],
                       "top_chord", f"top_{tag}_{i}")
        for i in range(n_panels + 1):
            add_member([xj[i], y, 0.0], [xj[i], y, top_z(xj[i])],
                       "vertical", f"vertical_{tag}_{i}")
        half = n_panels // 2
        for i in range(n_panels):
            if i < half:
                a = [xj[i], y, 0.0]
                b = [xj[i + 1], y, top_z(xj[i + 1])]
            else:
                a = [xj[i], y, top_z(xj[i])]
                b = [xj[i + 1], y, 0.0]
            add_member(a, b, "diagonal", f"diag_{tag}_{i}")

    # Interior longitudinal stringers: y = -0.75, 0, +0.75 m for width=3 m.
    for sy in ydeck[1:-1]:
        for i in range(n_panels):
            add_member([xj[i], sy, 0.0], [xj[i + 1], sy, 0.0],
                       "stringer", f"stringer_y{sy:+.3f}_{i}", deck_width=width / 4.0)

    # Floor beams at every panel line, split physically at each deck support line.
    for ix, x in enumerate(xj):
        for j in range(len(ydeck) - 1):
            add_member([x, ydeck[j], 0.0], [x, ydeck[j + 1], 0.0],
                       "floor_beam", f"floor_x{ix}_{j}")

    # Top cross-beams and one alternating diagonal in each top bay.
    for ix, x in enumerate(xj):
        add_member([x, -width / 2.0, top_z(x)], [x, width / 2.0, top_z(x)],
                   "top_cross", f"top_cross_{ix}")
    for i in range(n_panels):
        if i % 2 == 0:
            a = [xj[i], -width / 2.0, top_z(xj[i])]
            b = [xj[i + 1], width / 2.0, top_z(xj[i + 1])]
        else:
            a = [xj[i], width / 2.0, top_z(xj[i])]
            b = [xj[i + 1], -width / 2.0, top_z(xj[i + 1])]
        add_member(a, b, "top_brace", f"top_brace_{i}")

    n_members = len(physical_members)
    if radii is None:
        radii = np.full(n_members, 0.050)
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (n_members,):
        raise ValueError(f"radii must have shape ({n_members},), got {radii.shape}")

    elements: List[Tuple[int, int]] = []
    props_by_element: List[dict] = []
    element_member: List[int] = []
    deck_support_element_width: Dict[int, float] = {}

    for mid, m in enumerate(physical_members):
        pts = [m["a"] + (m["b"] - m["a"]) * (j / subdivisions) for j in range(subdivisions + 1)]
        ids = [reg.get(p) for p in pts]
        m["node_start"] = ids[0]
        m["node_end"] = ids[-1]
        m["radius"] = float(radii[mid])
        m["length"] = float(np.linalg.norm(m["b"] - m["a"]))
        for j in range(subdivisions):
            eid = len(elements)
            elements.append((ids[j], ids[j + 1]))
            props_by_element.append(chs_properties(radii[mid], E=E, nu=nu, rho=rho))
            element_member.append(mid)
            if m["deck_width"] is not None:
                deck_support_element_width[eid] = float(m["deck_width"])

    nodes = np.asarray(reg.coords, dtype=float)

    # Supports are applied along the entire deck end lines, as specified.
    fixed = []
    tol = 1e-9
    for nid, xyz in enumerate(nodes):
        x, y, z = xyz
        if abs(z) < tol and abs(x) < tol and -width / 2.0 - tol <= y <= width / 2.0 + tol:
            fixed.extend([6 * nid + 0, 6 * nid + 1, 6 * nid + 2])
        elif abs(z) < tol and abs(x - span) < tol and -width / 2.0 - tol <= y <= width / 2.0 + tol:
            fixed.extend([6 * nid + 1, 6 * nid + 2])

    return BridgeModel(
        nodes=nodes,
        elements=elements,
        props_by_element=props_by_element,
        element_member=np.asarray(element_member, dtype=int),
        members=physical_members,
        fixed_dofs=np.unique(np.asarray(fixed, dtype=int)),
        deck_support_element_width=deck_support_element_width,
        parameters={
            "span": span,
            "width": width,
            "H": H,
            "camber": camber,
            "n_panels": n_panels,
            "subdivisions": subdivisions,
            "deck_support_spacing": width / 4.0,
            "E": E,
            "nu": nu,
            "rho": rho,
        },
    )
