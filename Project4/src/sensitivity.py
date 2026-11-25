from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import spsolve

from .assembly import assemble, solve_static
from .bridge import build_footbridge
from .eigen import natural_frequencies, assemble_geometric_stiffness, linear_buckling
from .frame_element import local_stiffness, local_consistent_mass, local_geometric_stiffness_unit_compression
from .loads import assemble_bridge_loads, add_lumped_deck_mass, local_uniform_line_load
from .recovery import total_member_mass


STRESS_LIMIT = 53.3e6
DEFLECTION_LIMIT = 15.0 / 400.0
FREQUENCY_LIMIT = 3.0
BUCKLING_LIMIT = 3.0
KS_RHO = 60.0
DECK_MASS = 7850.0 * 0.008 * 15.0 * 3.0


@dataclass
class DesignAnalysis:
    radii: np.ndarray
    H: float
    camber: float
    model: object
    K: object
    Mstruct: object
    M: object
    cache: list
    F: np.ndarray
    element_loads_local: list
    u: np.ndarray
    reactions: np.ndarray
    free: np.ndarray
    vib: dict
    KG: object
    axial: np.ndarray
    buck: dict


def ks_value_weights(a, rho=KS_RHO):
    a = np.asarray(a, dtype=float)
    amax = float(np.max(a))
    z = np.exp(rho * (a - amax))
    s = float(np.sum(z))
    return amax + np.log(s) / rho, z / s


def _section_derivatives(props):
    r = props['r']
    return {
        'A': 2.0 * props['A'] / r,
        'Iy': 4.0 * props['Iy'] / r,
        'Iz': 4.0 * props['Iz'] / r,
        'J': 4.0 * props['J'] / r,
        'r': 1.0,
    }


def element_radius_derivatives(model, cache, e, gravity=9.81):
    """Return dK_e/dr, dM_e/dr, dF_e/dr for a child element.

    Only structural self weight depends on radius. D2 deck pressure/mass do not.
    """
    props = model.props_by_element[e]
    idx, T, L, _kl = cache[e]
    dp = _section_derivatives(props)
    dkl = local_stiffness(props['E'], props['G'], dp['A'], dp['Iy'], dp['Iz'], dp['J'], L)
    dml = local_consistent_mass(props['rho'], dp['A'], 0.0, 0.0, dp['J'], L)
    dke = T.T @ dkl @ T
    dme = T.T @ dml @ T

    R = T[:3, :3]
    dq_global = props['rho'] * dp['A'] * np.array([0.0, 0.0, -gravity])
    dfl = local_uniform_line_load(R @ dq_global, L)
    dfg = T.T @ dfl
    return idx, dke, dme, dfg, dkl, dfl


def analyze_design(radii, H=2.0, camber=0.0, n_modes=6):
    radii = np.asarray(radii, dtype=float)
    model = build_footbridge(radii=radii, H=float(H), camber=float(camber))
    K, Mstruct, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, eload, _ = assemble_bridge_loads(model, cache)
    u, reactions, free = solve_static(K, F, model.fixed_dofs)
    M, _ = add_lumped_deck_mass(Mstruct, model, cache)
    vib = natural_frequencies(K, M, model.fixed_dofs, n_modes=n_modes)
    KG, axial = assemble_geometric_stiffness(model, u, cache, eload)
    buck = linear_buckling(K, KG, model.fixed_dofs, n_modes=n_modes)
    return DesignAnalysis(radii, float(H), float(camber), model, K, Mstruct, M, cache, F,
                          eload, u, reactions, free, vib, KG, axial, buck)


def _smooth_stress_end(props, q, off):
    """Smooth project stress plus derivatives wrt q=[N,T,My,Mz] and explicit r.

    The smoothing constants are tiny in SI units and only remove the derivative cusp at zero.
    Values are numerically indistinguishable from the project formula at engineering stresses.
    """
    N, Tq, My, Mz = q[off+0], q[off+3], q[off+4], q[off+5]
    epsN, epsT, epsM, epsVM = 1e-6, 1e-9, 1e-9, 1e-3
    aN = np.sqrt(N*N + epsN*epsN)
    aT = np.sqrt(Tq*Tq + epsT*epsT)
    m = np.sqrt(My*My + Mz*Mz + epsM*epsM)
    A, I, J, r = props['A'], props['Iy'], props['J'], props['r']
    sigma = aN / A + m * r / I
    tau = aT * r / J
    vm = np.sqrt(sigma*sigma + 3.0*tau*tau + epsVM*epsVM)

    ds_dN = (N / aN) / A
    ds_dMy = (My / m) * r / I
    ds_dMz = (Mz / m) * r / I
    dt_dT = (Tq / aT) * r / J
    dvm_ds = sigma / vm
    dvm_dt = 3.0 * tau / vm
    grad_q = np.zeros(12)
    grad_q[off+0] = dvm_ds * ds_dN
    grad_q[off+3] = dvm_dt * dt_dT
    grad_q[off+4] = dvm_ds * ds_dMy
    grad_q[off+5] = dvm_ds * ds_dMz

    dp = _section_derivatives(props)
    ds_dr_exp = -aN * dp['A'] / A**2 + m * (1.0/I - r*dp['Iy']/I**2)
    dt_dr_exp = aT * (1.0/J - r*dp['J']/J**2)
    dvm_dr_exp = dvm_ds * ds_dr_exp + dvm_dt * dt_dr_exp
    return float(vm), grad_q, float(dvm_dr_exp)


def stress_ks(analysis, with_partials=False):
    vals = []
    records = []
    for e, item in enumerate(analysis.cache):
        idx, T, _L, kl = item
        ul = T @ analysis.u[idx]
        q = kl @ ul - analysis.element_loads_local[e]
        for off in (0, 6):
            vm, grad_q, dvm_dr_exp = _smooth_stress_end(analysis.model.props_by_element[e], q, off)
            vals.append(vm / STRESS_LIMIT)
            records.append((e, idx, T, kl, ul, q, grad_q, dvm_dr_exp))
    ks, w = ks_value_weights(vals)
    if not with_partials:
        return ks - 1.0, np.asarray(vals), w

    gu = np.zeros_like(analysis.u)
    gd_r = np.zeros(len(analysis.model.members))
    for wi, rec in zip(w, records):
        e, idx, T, kl, ul, q, grad_q, dvm_dr_exp = rec
        # q = kl T u - fl
        gu[idx] += wi / STRESS_LIMIT * (T.T @ (kl.T @ grad_q))
        _idx, _dke, _dme, _dfg, dkl, dfl = element_radius_derivatives(analysis.model, analysis.cache, e)
        dqdr = dkl @ ul - dfl
        dvm_dr = float(grad_q @ dqdr + dvm_dr_exp)
        gd_r[analysis.model.element_member[e]] += wi * dvm_dr / STRESS_LIMIT
    return ks - 1.0, np.asarray(vals), w, gu, gd_r


def deflection_ks(analysis, with_partials=False):
    deck_nodes = np.array([i for i,x in enumerate(analysis.model.nodes) if abs(x[2]) < 1e-10], dtype=int)
    dofs = 6*deck_nodes + 2
    vals = -analysis.u[dofs] / DEFLECTION_LIMIT
    ks, w = ks_value_weights(vals)
    if not with_partials:
        return ks - 1.0, vals, w
    gu = np.zeros_like(analysis.u)
    gu[dofs] = -w / DEFLECTION_LIMIT
    return ks - 1.0, vals, w, gu


def vibration_ks(analysis, with_partials=False):
    f = np.asarray(analysis.vib['frequencies_hz'], float)
    lam = np.asarray(analysis.vib['eigenvalues'], float)
    modes = np.asarray(analysis.vib['modes'], float).copy()
    nm = min(6, len(f))
    f, lam, modes = f[:nm], lam[:nm], modes[:, :nm]
    # Normalize explicitly to phi^T M phi = 1.
    for i in range(nm):
        nrm = np.sqrt(modes[:,i] @ (analysis.M @ modes[:,i]))
        modes[:,i] /= nrm
    a = FREQUENCY_LIMIT / f
    ks, w = ks_value_weights(a)
    if not with_partials:
        return ks - 1.0, a, w

    gd_r = np.zeros(len(analysis.model.members))
    for e in range(len(analysis.model.elements)):
        idx, dke, dme, _dfg, _dkl, _dfl = element_radius_derivatives(analysis.model, analysis.cache, e)
        mid = analysis.model.element_member[e]
        for i in range(nm):
            ph = modes[idx, i]
            dlam = ph @ (dke - lam[i]*dme) @ ph
            df = dlam / (4.0*np.pi*np.sqrt(lam[i]))
            da = -FREQUENCY_LIMIT * df / f[i]**2
            gd_r[mid] += w[i] * da
    return ks - 1.0, a, w, gd_r, modes


def _axial_force_gradient_u(analysis, e):
    idx, T, _L, kl = analysis.cache[e]
    sel = np.zeros(12); sel[6] = 0.5; sel[0] = -0.5
    return idx, T.T @ (kl.T @ sel)


def buckling_ks(analysis, with_partials=False):
    mu = np.asarray(analysis.buck['mu'], float)
    modes = np.asarray(analysis.buck['modes'], float).copy()
    nm = min(6, len(mu))
    mu, modes = mu[:nm], modes[:, :nm]
    for i in range(nm):
        nrm = np.sqrt(modes[:,i] @ (analysis.K @ modes[:,i]))
        modes[:,i] /= nrm
    a = BUCKLING_LIMIT * mu  # alpha>=3 <=> 3*mu <= 1
    ks, w = ks_value_weights(a)
    if not with_partials:
        return ks - 1.0, a, w

    gu = np.zeros_like(analysis.u)
    gd_r = np.zeros(len(analysis.model.members))

    # Precompute c[e,i] = psi_e^T G_unit_global psi_e.
    c = np.zeros((len(analysis.model.elements), nm))
    for e, item in enumerate(analysis.cache):
        idx, T, L, _kl = item
        Gg = T.T @ local_geometric_stiffness_unit_compression(L) @ T
        for i in range(nm):
            pe = modes[idx, i]
            c[e,i] = pe @ Gg @ pe
        idxg, dNdu = _axial_force_gradient_u(analysis, e)
        coeff = -BUCKLING_LIMIT * float(np.dot(w, c[e,:]))
        gu[idxg] += coeff * dNdu

    # Direct radius partial at fixed u; G_unit has no radius dependence.
    for e, item in enumerate(analysis.cache):
        idx, T, _L, kl = item
        ul = T @ analysis.u[idx]
        _idx, dke, _dme, _dfg, dkl, dfl = element_radius_derivatives(analysis.model, analysis.cache, e)
        dqdr = dkl @ ul - dfl
        dNdr = 0.5 * (dqdr[6] - dqdr[0])
        mid = analysis.model.element_member[e]
        for i in range(nm):
            pe = modes[idx, i]
            partial_mu = -c[e,i] * dNdr - mu[i] * (pe @ dke @ pe)
            gd_r[mid] += w[i] * BUCKLING_LIMIT * partial_mu
    return ks - 1.0, a, w, gu, gd_r, modes, mu


def solve_adjoint(K, fixed_dofs, gu):
    n = K.shape[0]
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)
    lam = np.zeros(n)
    lam[free] = spsolve(K[free][:,free].T, -np.asarray(gu)[free])
    return lam


def residual_radius_term(analysis, lam, e):
    idx, dke, _dme, dfg, _dkl, _dfl = element_radius_derivatives(analysis.model, analysis.cache, e)
    return float(lam[idx] @ (dke @ analysis.u[idx] - dfg))


def _shape_perturbed_analysis_parts(analysis, name, delta):
    H = analysis.H + (delta if name == 'H' else 0.0)
    camber = analysis.camber + (delta if name == 'camber' else 0.0)
    model = build_footbridge(radii=analysis.radii, H=H, camber=camber)
    K, Mstruct, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, eload, _ = assemble_bridge_loads(model, cache)
    M, _ = add_lumped_deck_mass(Mstruct, model, cache)
    return model, K, M, cache, F, eload


def _shape_step(analysis, name):
    scale = max(abs(analysis.H if name == 'H' else analysis.camber), 1.0)
    return 2e-6 * scale


def response_values(analysis):
    sg, sv, _ = stress_ks(analysis)
    dg, dv, _ = deflection_ks(analysis)
    vg, va, _ = vibration_ks(analysis)
    bg, ba, _ = buckling_ks(analysis)
    member_mass = total_member_mass(analysis.model)
    true_max_stress = float(np.max(sv) * STRESS_LIMIT)
    true_max_down = float(np.max(dv) * DEFLECTION_LIMIT)
    return {
        'mass_total_kg': member_mass + DECK_MASS,
        'mass_members_kg': member_mass,
        'stress_ks': float(sg),
        'deflection_ks': float(dg),
        'vibration_ks': float(vg),
        'buckling_ks': float(bg),
        'true_max_stress_pa': true_max_stress,
        'true_max_down_m': true_max_down,
        'lowest_frequency_hz': float(analysis.vib['frequencies_hz'][0]),
        'buckling_factor': float(analysis.buck['load_factors'][0]),
    }


def reduced_gradients(analysis):
    """Reduced gradients wrt physical d=[all member radii, H, camber].

    Geometry-variable operator partials are centered differences at fixed state.  They are
    assembled once per +/- perturbation and reused by every response; no perturbed PDE solve
    appears in the reduced-gradient calculation.
    """
    nmembers = len(analysis.model.members)
    nd = nmembers + 2
    out = {}

    # Cache the four perturbed operator assemblies once.  This is important for optimization:
    # the shape partials are shared by mass, stress, deflection, vibration, and buckling.
    shape_parts = {}
    for name in ('H', 'camber'):
        h = _shape_step(analysis, name)
        shape_parts[name] = (
            h,
            _shape_perturbed_analysis_parts(analysis, name, +h),
            _shape_perturbed_analysis_parts(analysis, name, -h),
        )

    # Mass: exact radius derivative, centered geometric partial for the two shape variables.
    gm = np.zeros(nd)
    rho = analysis.model.parameters['rho']
    for mid,m in enumerate(analysis.model.members):
        r = m['radius']
        A = np.pi*(r*r - (0.9*r)**2)
        gm[mid] = rho * (2*A/r) * m['length']
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        mp = pp[0]; mm = pm[0]
        gm[nmembers+j] = (total_member_mass(mp)-total_member_mass(mm))/(2*h)
    out['mass_total_kg'] = gm

    # Stress KS: one adjoint solve.
    _g, _vals, _w, gu, gd = stress_ks(analysis, with_partials=True)
    lam = solve_adjoint(analysis.K, analysis.model.fixed_dofs, gu)
    gs = np.zeros(nd); gs[:nmembers] = gd
    for e in range(len(analysis.model.elements)):
        gs[analysis.model.element_member[e]] += residual_radius_term(analysis, lam, e)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        modelp,Kp,_Mp,cachep,Fp,eloadp = pp
        modelm,Km,_Mm,cachem,Fm,eloadm = pm
        # Partial response at fixed global state u.
        ap = DesignAnalysis(analysis.radii, modelp.parameters['H'], modelp.parameters['camber'], modelp, Kp, None, None, cachep, Fp, eloadp, analysis.u, None, analysis.free, analysis.vib, None, None, analysis.buck)
        am = DesignAnalysis(analysis.radii, modelm.parameters['H'], modelm.parameters['camber'], modelm, Km, None, None, cachem, Fm, eloadm, analysis.u, None, analysis.free, analysis.vib, None, None, analysis.buck)
        gp = stress_ks(ap)[0]; gm_ = stress_ks(am)[0]
        direct = (gp-gm_)/(2*h)
        Kdu_minus_Fd = ((Kp-Km) @ analysis.u - (Fp-Fm)) / (2*h)
        gs[nmembers+j] = direct + lam @ Kdu_minus_Fd
    out['stress_ks'] = gs

    # Deflection KS: one adjoint solve, no explicit d-dependence.
    _g, _vals, _w, gu = deflection_ks(analysis, with_partials=True)
    lam = solve_adjoint(analysis.K, analysis.model.fixed_dofs, gu)
    gdfl = np.zeros(nd)
    for e in range(len(analysis.model.elements)):
        gdfl[analysis.model.element_member[e]] += residual_radius_term(analysis, lam, e)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        _modelp,Kp,_Mp,_cachep,Fp,_eloadp = pp
        _modelm,Km,_Mm,_cachem,Fm,_eloadm = pm
        Kdu_minus_Fd = ((Kp-Km) @ analysis.u - (Fp-Fm)) / (2*h)
        gdfl[nmembers+j] = lam @ Kdu_minus_Fd
    out['deflection_ks'] = gdfl

    # Vibration KS: eigenvalue sensitivity, no state adjoint.
    _g, _a, wv, gv_r, modes = vibration_ks(analysis, with_partials=True)
    gv = np.zeros(nd); gv[:nmembers] = gv_r
    f = np.asarray(analysis.vib['frequencies_hz'][:len(wv)], float)
    eig = np.asarray(analysis.vib['eigenvalues'][:len(wv)], float)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        _modelp,Kp,Mp,_cachep,_Fp,_eloadp = pp
        _modelm,Km,Mm,_cachem,_Fm,_eloadm = pm
        Kd = (Kp-Km)/(2*h); Md=(Mp-Mm)/(2*h)
        val = 0.0
        for i in range(len(wv)):
            ph = modes[:,i]
            dlam = ph @ (Kd @ ph) - eig[i]*(ph @ (Md @ ph))
            df = dlam/(4*np.pi*np.sqrt(eig[i]))
            da = -FREQUENCY_LIMIT*df/f[i]**2
            val += wv[i]*da
        gv[nmembers+j] = val
    out['vibration_ks'] = gv

    # Buckling KS: aggregate all eigenvalue state terms first, then one adjoint.
    _g, _a, wb, gu, gb_r, modesb, mub = buckling_ks(analysis, with_partials=True)
    lamb = solve_adjoint(analysis.K, analysis.model.fixed_dofs, gu)
    gb = np.zeros(nd); gb[:nmembers] = gb_r
    for e in range(len(analysis.model.elements)):
        gb[analysis.model.element_member[e]] += residual_radius_term(analysis, lamb, e)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        modelp,Kp,_Mp,cachep,Fp,eloadp = pp
        modelm,Km,_Mm,cachem,Fm,eloadm = pm
        # Direct eigenvalue partial at fixed u, with base eigenvectors/mu/KS weights fixed.
        KGp,_ = assemble_geometric_stiffness(modelp, analysis.u, cachep, eloadp)
        KGm,_ = assemble_geometric_stiffness(modelm, analysis.u, cachem, eloadm)
        sp=0.0; sm=0.0
        for i in range(len(wb)):
            psi=modesb[:,i]
            sp += wb[i]*BUCKLING_LIMIT*(psi @ ((-KGp - mub[i]*Kp) @ psi))
            sm += wb[i]*BUCKLING_LIMIT*(psi @ ((-KGm - mub[i]*Km) @ psi))
        direct=(sp-sm)/(2*h)
        Kdu_minus_Fd=((Kp-Km)@analysis.u-(Fp-Fm))/(2*h)
        gb[nmembers+j]=direct+lamb@Kdu_minus_Fd
    out['buckling_ks'] = gb

    return out

def design_scales(nmembers):
    return np.r_[np.full(nmembers, 0.05), 2.0, 0.5]


def unpack_scaled_design(x, nmembers):
    s = design_scales(nmembers)
    d = np.asarray(x,float)*s
    return d[:nmembers], d[nmembers], d[nmembers+1]


def pack_scaled_design(radii, H, camber):
    d=np.r_[np.asarray(radii,float),H,camber]
    return d/design_scales(len(radii))


def analyze_static_design(radii, H=2.0, camber=0.0):
    """Static-only bridge state for inexpensive optimization line-search evaluations."""
    radii = np.asarray(radii, dtype=float)
    model = build_footbridge(radii=radii, H=float(H), camber=float(camber))
    K, Mstruct, cache = assemble(model.nodes, model.elements, model.props_by_element)
    F, eload, _ = assemble_bridge_loads(model, cache)
    u, reactions, free = solve_static(K, F, model.fixed_dofs)
    return DesignAnalysis(radii, float(H), float(camber), model, K, Mstruct, None, cache, F,
                          eload, u, reactions, free, None, None, None, None)


def static_response_values(analysis):
    sg, sv, _ = stress_ks(analysis)
    dg, dv, _ = deflection_ks(analysis)
    member_mass = total_member_mass(analysis.model)
    return {
        'mass_total_kg': member_mass + DECK_MASS,
        'mass_members_kg': member_mass,
        'stress_ks': float(sg),
        'deflection_ks': float(dg),
        'true_max_stress_pa': float(np.max(sv) * STRESS_LIMIT),
        'true_max_down_m': float(np.max(dv) * DEFLECTION_LIMIT),
    }


def reduced_static_gradients(analysis):
    """Reduced gradients for mass, stress KS, and deflection KS only.

    This is numerically identical to the corresponding entries of reduced_gradients(), but
    avoids vibration/buckling work so SLSQP line-search staging is inexpensive.
    """
    nmembers = len(analysis.model.members)
    nd = nmembers + 2
    out = {}
    shape_parts = {}
    for name in ('H', 'camber'):
        h = _shape_step(analysis, name)
        shape_parts[name] = (
            h,
            _shape_perturbed_analysis_parts(analysis, name, +h),
            _shape_perturbed_analysis_parts(analysis, name, -h),
        )

    gm = np.zeros(nd)
    rho = analysis.model.parameters['rho']
    for mid,m in enumerate(analysis.model.members):
        r = m['radius']; A = np.pi*(r*r - (0.9*r)**2)
        gm[mid] = rho * (2*A/r) * m['length']
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        gm[nmembers+j] = (total_member_mass(pp[0])-total_member_mass(pm[0]))/(2*h)
    out['mass_total_kg'] = gm

    _g, _vals, _w, gu, gd = stress_ks(analysis, with_partials=True)
    lam = solve_adjoint(analysis.K, analysis.model.fixed_dofs, gu)
    gs = np.zeros(nd); gs[:nmembers] = gd
    for e in range(len(analysis.model.elements)):
        gs[analysis.model.element_member[e]] += residual_radius_term(analysis, lam, e)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        modelp,Kp,_Mp,cachep,Fp,eloadp = pp
        modelm,Km,_Mm,cachem,Fm,eloadm = pm
        ap = DesignAnalysis(analysis.radii, modelp.parameters['H'], modelp.parameters['camber'], modelp, Kp, None, None, cachep, Fp, eloadp, analysis.u, None, analysis.free, None, None, None, None)
        am = DesignAnalysis(analysis.radii, modelm.parameters['H'], modelm.parameters['camber'], modelm, Km, None, None, cachem, Fm, eloadm, analysis.u, None, analysis.free, None, None, None, None)
        direct = (stress_ks(ap)[0]-stress_ks(am)[0])/(2*h)
        gs[nmembers+j] = direct + lam @ (((Kp-Km)@analysis.u-(Fp-Fm))/(2*h))
    out['stress_ks'] = gs

    _g, _vals, _w, gu = deflection_ks(analysis, with_partials=True)
    lam = solve_adjoint(analysis.K, analysis.model.fixed_dofs, gu)
    gdfl = np.zeros(nd)
    for e in range(len(analysis.model.elements)):
        gdfl[analysis.model.element_member[e]] += residual_radius_term(analysis, lam, e)
    for j,name in enumerate(('H','camber')):
        h, pp, pm = shape_parts[name]
        _modelp,Kp,_Mp,_cachep,Fp,_eloadp = pp
        _modelm,Km,_Mm,_cachem,Fm,_eloadm = pm
        gdfl[nmembers+j] = lam @ (((Kp-Km)@analysis.u-(Fp-Fm))/(2*h))
    out['deflection_ks'] = gdfl
    return out
