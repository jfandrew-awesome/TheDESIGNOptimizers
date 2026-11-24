from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh

from .frame_element import local_geometric_stiffness_unit_compression


def supported_submatrices(K, M, fixed_dofs):
    """Return free DOFs and the supported stiffness/mass matrices."""
    n = K.shape[0]
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)
    return free, K[free][:, free].tocsr(), M[free][:, free].tocsr()


def natural_frequencies(K, M, fixed_dofs, n_modes=10):
    """Lowest supported natural frequencies for K phi = omega^2 M phi.

    Eigenvectors are returned in full global-DOF coordinates and are M-normalized
    by scipy's symmetric generalized eigensolver.
    """
    free, Kf, Mf = supported_submatrices(K, M, fixed_dofs)
    k = min(int(n_modes), Kf.shape[0] - 1)
    if k < 1:
        raise ValueError("not enough free DOFs for an eigenproblem")

    # Shift-invert about zero is robust for the lowest positive modes of a
    # supported structure with positive-definite K and M.
    lam, vec = eigsh(Kf, M=Mf, k=k, sigma=0.0, which="LM")
    order = np.argsort(lam)
    lam = np.asarray(lam[order], float)
    vec = np.asarray(vec[:, order], float)
    keep = lam > max(1e-12, 1e-10 * np.max(np.abs(lam)))
    lam = lam[keep]
    vec = vec[:, keep]
    omega = np.sqrt(lam)
    freq = omega / (2.0 * np.pi)

    modes = np.zeros((K.shape[0], len(lam)))
    modes[free, :] = vec
    return {
        "eigenvalues": lam,
        "omega": omega,
        "frequencies_hz": freq,
        "modes": modes,
        "free_dofs": free,
    }


def element_axial_force_tension_positive(u, cache_item, element_load_local):
    """Average element axial force, positive in tension and negative in compression.

    With q = k u - f as the local element end-force vector, a constant tensile
    force has q_x(left)=-N and q_x(right)=+N. Averaging the two end estimates
    also handles the small axial component of member self-weight on inclined members.
    """
    idx, T, _L, kl = cache_item
    ul = T @ np.asarray(u)[idx]
    q = kl @ ul - np.asarray(element_load_local)
    return 0.5 * (q[6] - q[0])


def assemble_geometric_stiffness(model, u, cache, element_loads_local):
    """Assemble K_G from the current static state.

    Sign convention follows the project notes: K + alpha K_G becomes singular.
    Tension gives positive geometric stiffness; compression gives negative stiffness.
    """
    n = 6 * len(model.nodes)
    KG = lil_matrix((n, n))
    axial = np.zeros(len(model.elements))

    for e, item in enumerate(cache):
        idx, T, L, _kl = item
        N = element_axial_force_tension_positive(u, item, element_loads_local[e])
        axial[e] = N
        # This matrix is the conventional positive geometric matrix per unit
        # tensile force. Therefore signed N directly gives tension stiffening
        # and compression softening.
        gl = local_geometric_stiffness_unit_compression(L)
        ge = T.T @ (N * gl) @ T
        for a, I in enumerate(idx):
            for b, J in enumerate(idx):
                KG[I, J] += ge[a, b]

    return csr_matrix(KG), axial


def linear_buckling(K, KG, fixed_dofs, n_modes=10):
    """Solve (-K_G) psi = mu K psi, alpha = 1/mu.

    The largest positive mu is the smallest positive buckling load factor.
    This form keeps the generalized mass matrix K positive definite, allowing
    a symmetric sparse eigensolve even though -K_G is indefinite.
    """
    n = K.shape[0]
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)
    Kf = K[free][:, free].tocsr()
    Af = (-KG[free][:, free]).tocsr()
    k = min(int(n_modes), Kf.shape[0] - 1)
    if k < 1:
        raise ValueError("not enough free DOFs for buckling eigenproblem")

    mu, vec = eigsh(Af, M=Kf, k=k, which="LA")
    order = np.argsort(mu)[::-1]
    mu = np.asarray(mu[order], float)
    vec = np.asarray(vec[:, order], float)
    positive = mu > max(1e-14, 1e-10 * np.max(np.abs(mu)))
    mu_pos = mu[positive]
    vec_pos = vec[:, positive]
    if len(mu_pos) == 0:
        raise RuntimeError("no positive buckling eigenvalue found")

    alpha = 1.0 / mu_pos
    modes = np.zeros((n, len(mu_pos)))
    modes[free, :] = vec_pos
    return {
        "mu": mu_pos,
        "load_factors": alpha,
        "modes": modes,
        "free_dofs": free,
        "raw_mu": mu,
    }
