import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembly import assemble, solve_static
from src.loads import local_uniform_line_load
from src.sections import chs_properties


def rel(a, b):
    return abs(a-b) / max(abs(b), 1e-30)


def main():
    # Cantilever along global/local x under a uniform downward (-z) load.
    L = 2.4
    q = 1800.0
    p = chs_properties(0.05)
    nodes = np.array([[0., 0., 0.], [L, 0., 0.]])
    K, M, cache = assemble(nodes, [(0, 1)], [p])
    idx, T, Le, kl = cache[0]

    q_global = np.array([0.0, 0.0, -q])
    R = T[:3, :3]
    fl = local_uniform_line_load(R @ q_global, Le)
    F = np.zeros(12)
    F[idx] += T.T @ fl
    u, reactions, _ = solve_static(K, F, np.arange(6))

    # Exact Euler-Bernoulli cantilever response under UDL.
    exact_tip = -q * L**4 / (8.0 * p['E'] * p['Iy'])
    exact_root_moment_mag = q * L**2 / 2.0
    exact_root_shear_mag = q * L

    ul = T @ u[idx]
    end_force = kl @ ul - fl
    tip = u[8]
    root_shear = abs(end_force[2])
    root_moment = abs(end_force[4])

    assert rel(tip, exact_tip) < 1e-11, (tip, exact_tip)
    assert rel(root_shear, exact_root_shear_mag) < 1e-12
    assert rel(root_moment, exact_root_moment_mag) < 1e-12

    exact_sigma = exact_root_moment_mag * p['r'] / p['Iy']
    recovered_sigma = root_moment * p['r'] / p['Iy']
    assert rel(recovered_sigma, exact_sigma) < 1e-12

    print('Distributed-load / recovery benchmark: PASS')
    print(f'  tip z = {tip:.12e} m, exact {exact_tip:.12e}, relerr {rel(tip, exact_tip):.3e}')
    print(f'  root shear = {root_shear:.6f} N, exact {exact_root_shear_mag:.6f}')
    print(f'  root bending moment = {root_moment:.6f} N*m, exact {exact_root_moment_mag:.6f}')
    print(f'  root bending stress = {recovered_sigma/1e6:.6f} MPa')


if __name__ == '__main__':
    main()
