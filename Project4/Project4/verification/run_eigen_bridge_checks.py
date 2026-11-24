"""Milestone 3 verification: element eigen benchmarks + bridge eigen residuals."""
import pathlib
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import lil_matrix, csr_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, dofs_for_element, solve_static
from src.bridge import build_footbridge
from src.eigen import natural_frequencies, assemble_geometric_stiffness, linear_buckling
from src.frame_element import element_matrices, local_geometric_stiffness_unit_compression
from src.loads import assemble_bridge_loads, add_lumped_deck_mass
from src.sections import chs_properties


def beam_frequency(n):
    L = 4.0
    p = chs_properties(0.05)
    nodes = np.c_[np.linspace(0.0, L, n + 1), np.zeros(n + 1), np.zeros(n + 1)]
    elems = [(i, i + 1) for i in range(n)]
    K, M, _ = assemble(nodes, elems, [p] * n)
    fixed = []
    for i in range(n + 1):
        fixed += [6*i+0, 6*i+2, 6*i+3, 6*i+4]
    fixed += [1, 6*n+1]
    ans = natural_frequencies(K, M, fixed, n_modes=1)
    exact = np.pi / (2 * L**2) * np.sqrt(p['E'] * p['Iz'] / (p['rho'] * p['A']))
    return ans['frequencies_hz'][0], exact


def euler_column(n):
    L = 3.0
    p = chs_properties(0.05)
    nn = n + 1
    K = lil_matrix((6*nn, 6*nn))
    KG = lil_matrix((6*nn, 6*nn))
    x = np.linspace(0.0, L, nn)
    # Use a unit compressive state: N=-1 N, so KG=-G_unit.
    for e in range(n):
        ke, _, T, Le, _ = element_matrices([x[e],0,0], [x[e+1],0,0], p)
        ge = T.T @ (-local_geometric_stiffness_unit_compression(Le)) @ T
        ids = dofs_for_element(e, e+1)
        for a,I in enumerate(ids):
            for b,J in enumerate(ids):
                K[I,J] += ke[a,b]
                KG[I,J] += ge[a,b]
    # Isolate one pinned-pinned bending plane to avoid duplicate eigenvalues.
    active = []
    for i in range(nn):
        active += [6*i+1, 6*i+5]
    fixed_in_active = [1, 6*n+1]
    free_active = np.setdiff1d(np.asarray(active), fixed_in_active)
    Kd = csr_matrix(K)[free_active][:, free_active]
    KGd = csr_matrix(KG)[free_active][:, free_active]
    # No additional supports are needed after reducing to the active DOFs.
    ans = linear_buckling(Kd, KGd, [], n_modes=1)
    exact = np.pi**2 * p['E'] * p['Iz'] / L**2
    return ans['load_factors'][0], exact


def bridge_checks():
    model = build_footbridge()
    K, M, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, fl, _ = assemble_bridge_loads(model, cache)
    u, _, free = solve_static(K, F, model.fixed_dofs)
    M, _ = add_lumped_deck_mass(M, model, cache)
    vib = natural_frequencies(K, M, model.fixed_dofs, n_modes=6)
    KG, axial = assemble_geometric_stiffness(model, u, cache, fl)
    buck = linear_buckling(K, KG, model.fixed_dofs, n_modes=6)

    phi = vib['modes'][free, 0]
    rv = K[free][:,free] @ phi - vib['eigenvalues'][0] * (M[free][:,free] @ phi)
    vib_rel = np.linalg.norm(rv) / (np.linalg.norm(K[free][:,free] @ phi) + np.linalg.norm(vib['eigenvalues'][0]*(M[free][:,free] @ phi)))
    psi = buck['modes'][free, 0]
    rb = (-KG[free][:,free]) @ psi - buck['mu'][0] * (K[free][:,free] @ psi)
    buck_rel = np.linalg.norm(rb) / (np.linalg.norm((-KG[free][:,free]) @ psi) + np.linalg.norm(buck['mu'][0]*(K[free][:,free] @ psi)))
    return vib, buck, axial, vib_rel, buck_rel


if __name__ == '__main__':
    print('V3 frequency convergence')
    last = None
    for n in [1,2,4,8,16]:
        got, exact = beam_frequency(n)
        err = abs(got-exact)/exact
        print(f'  n={n:2d}: got={got:.9f} Hz exact={exact:.9f} relerr={err:.3e}')
        last = err
    assert last < 1e-5

    print('V4 Euler buckling convergence')
    last = None
    for n in [1,2,4,8]:
        got, exact = euler_column(n)
        err = abs(got-exact)/exact
        print(f'  n={n:2d}: got={got:.6f} N exact={exact:.6f} relerr={err:.3e}')
        last = err
    assert last < 5e-5

    vib, buck, axial, rv, rb = bridge_checks()
    print('Bridge eigen checks')
    print('  frequencies Hz:', np.array2string(vib['frequencies_hz'], precision=6))
    print('  buckling factors:', np.array2string(buck['load_factors'], precision=6))
    print(f'  vibration residual={rv:.3e}')
    print(f'  buckling residual ={rb:.3e}')
    print(f'  axial range={axial.min()/1e3:.3f} .. {axial.max()/1e3:.3f} kN')
    assert np.all(np.diff(vib['frequencies_hz']) >= -1e-9)
    assert np.all(buck['load_factors'] > 0)
    assert rv < 1e-8
    assert rb < 1e-8
