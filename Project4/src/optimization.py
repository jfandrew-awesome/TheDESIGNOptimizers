from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from .bridge import build_footbridge
from .sensitivity import (
    analyze_design,
    response_values,
    reduced_gradients,
    pack_scaled_design,
    unpack_scaled_design,
    design_scales,
    STRESS_LIMIT,
    DEFLECTION_LIMIT,
    FREQUENCY_LIMIT,
    BUCKLING_LIMIT,
    DECK_MASS,
)

SLENDERNESS_LIMIT = 200.0
RG_OVER_R = 0.673


@dataclass
class EvaluationCache:
    nmembers: int
    include_buckling: bool = True
    x: np.ndarray | None = None
    analysis: object | None = None
    values: dict | None = None
    grads: dict | None = None
    n_analyses: int = 0
    n_gradients: int = 0
    analysis_time: float = 0.0
    gradient_time: float = 0.0

    def _same(self, x):
        return self.x is not None and np.array_equal(np.asarray(x), self.x)

    def ensure_analysis(self, x):
        x = np.asarray(x, float)
        if not self._same(x):
            radii, H, camber = unpack_scaled_design(x, self.nmembers)
            t0 = time.perf_counter()
            a = analyze_design(radii, H, camber)
            dt = time.perf_counter() - t0
            self.x = x.copy()
            self.analysis = a
            self.values = response_values(a)
            self.grads = None
            self.n_analyses += 1
            self.analysis_time += dt
        return self.analysis, self.values

    def ensure_gradients(self, x):
        self.ensure_analysis(x)
        if self.grads is None:
            t0 = time.perf_counter()
            self.grads = reduced_gradients(self.analysis)
            self.gradient_time += time.perf_counter() - t0
            self.n_gradients += 1
        return self.grads


def slenderness_values(analysis):
    vals = np.empty(len(analysis.model.members))
    for i, m in enumerate(analysis.model.members):
        vals[i] = m['length'] / (RG_OVER_R * m['radius']) / SLENDERNESS_LIMIT - 1.0
    return vals


def slenderness_jacobian_scaled(analysis):
    """Jacobian of g=L/(0.673 r)/200-1 wrt scaled x.

    Radius derivatives are analytical.  The two explicit geometry derivatives are centered
    differences of physical member lengths only; no PDE solve is involved.
    """
    nm = len(analysis.model.members)
    J = np.zeros((nm, nm + 2))
    scales = design_scales(nm)
    g = slenderness_values(analysis)
    for i, m in enumerate(analysis.model.members):
        # d g_i / d r_i = -(g_i+1)/r_i
        J[i, i] = -(g[i] + 1.0) / m['radius'] * scales[i]

    for j, name in enumerate(('H', 'camber')):
        base = analysis.H if name == 'H' else analysis.camber
        h = 2e-6 * max(abs(base), 1.0)
        Hp = analysis.H + (h if name == 'H' else 0.0)
        Hm = analysis.H - (h if name == 'H' else 0.0)
        cp = analysis.camber + (h if name == 'camber' else 0.0)
        cm = analysis.camber - (h if name == 'camber' else 0.0)
        mp = build_footbridge(radii=analysis.radii, H=Hp, camber=cp)
        mm = build_footbridge(radii=analysis.radii, H=Hm, camber=cm)
        for i in range(nm):
            dL = (mp.members[i]['length'] - mm.members[i]['length']) / (2*h)
            dg = dL / (RG_OVER_R * analysis.radii[i]) / SLENDERNESS_LIMIT
            J[i, nm+j] = dg * scales[nm+j]
    return J



def slenderness_ks(analysis, rho=300.0):
    """Conservative KS upper bound of member slenderness ratios L/(rg*200)."""
    g = slenderness_values(analysis)
    ratio = g + 1.0
    amax = float(np.max(ratio))
    z = np.exp(rho*(ratio-amax))
    w = z/np.sum(z)
    ks = amax + np.log(np.sum(z))/rho
    return float(ks-1.0), w

def slenderness_ks_jacobian_scaled(analysis, rho=300.0):
    _g,w=slenderness_ks(analysis,rho=rho)
    J=slenderness_jacobian_scaled(analysis)
    return w @ J

def default_scaled_bounds(nmembers):
    """Bounds chosen to stay within the assignment's radius limits and sane bridge shapes."""
    scales = design_scales(nmembers)
    lo = np.r_[np.full(nmembers, 0.010), 1.0, 0.0] / scales
    hi = np.r_[np.full(nmembers, 0.150), 4.0, 1.5] / scales
    return list(zip(lo, hi))


def _constraint_names(nmembers, include_buckling=True):
    names = ['R2_fatigue_KS', 'R3_deflection_KS', 'R4_vibration_KS']
    if include_buckling:
        names.append('R5_buckling_KS')
    names.append('R6_slenderness_KS')
    return names

def optimize_bridge(
    initial_r=0.0755,
    initial_H=3.0,
    initial_camber=0.0,
    include_buckling=True,
    maxiter=120,
    ftol=1e-7,
    disp=True,
):
    base = build_footbridge()
    nm = len(base.members)
    r0 = np.full(nm, float(initial_r)) if np.isscalar(initial_r) else np.asarray(initial_r, float)
    x0 = pack_scaled_design(r0, initial_H, initial_camber)
    bounds = default_scaled_bounds(nm)
    cache = EvaluationCache(nm, include_buckling=include_buckling)
    a0, v0 = cache.ensure_analysis(x0)
    mass_scale = float(v0['mass_total_kg'])
    scales = design_scales(nm)
    history = []

    pde_keys = ['stress_ks', 'deflection_ks', 'vibration_ks']
    if include_buckling:
        pde_keys.append('buckling_ks')

    def objective(x):
        _a, vals = cache.ensure_analysis(x)
        return vals['mass_total_kg'] / mass_scale

    def objective_jac(x):
        gg = cache.ensure_gradients(x)['mass_total_kg']
        return gg * scales / mass_scale

    def cons_fun(x):
        a, vals = cache.ensure_analysis(x)
        # scipy SLSQP inequality convention is c(x) >= 0.
        cpde = np.array([-vals[k] for k in pde_keys], float)
        csl = -slenderness_ks(a)[0]
        return np.r_[cpde, csl]

    def cons_jac(x):
        a, _vals = cache.ensure_analysis(x)
        grads = cache.ensure_gradients(x)
        rows = [-(grads[k] * scales) for k in pde_keys]
        Jsl = -slenderness_ks_jacobian_scaled(a)
        return np.vstack(rows + [Jsl])

    def callback(xk):
        a, vals = cache.ensure_analysis(xk)
        sl = slenderness_values(a)
        rec = {
            'iteration': len(history),
            'mass_total_kg': float(vals['mass_total_kg']),
            'mass_members_kg': float(vals['mass_members_kg']),
            'stress_ks': float(vals['stress_ks']),
            'deflection_ks': float(vals['deflection_ks']),
            'vibration_ks': float(vals['vibration_ks']),
            'buckling_ks': float(vals['buckling_ks']),
            'true_max_stress_mpa': float(vals['true_max_stress_pa']/1e6),
            'true_max_deflection_mm': float(vals['true_max_down_m']*1e3),
            'lowest_frequency_hz': float(vals['lowest_frequency_hz']),
            'buckling_factor': float(vals['buckling_factor']),
            'max_slenderness': float((sl.max()+1.0)*SLENDERNESS_LIMIT),
            'H_m': float(a.H),
            'camber_m': float(a.camber),
            'min_radius_mm': float(np.min(a.radii)*1e3),
            'max_radius_mm': float(np.max(a.radii)*1e3),
            'max_constraint_violation': float(max(0.0, -np.min(cons_fun(xk)))),
        }
        history.append(rec)
        if disp:
            print(
                f"iter {rec['iteration']:3d}: mass={rec['mass_total_kg']:8.1f} kg  "
                f"stressKS={rec['stress_ks']:+.3e}  deflKS={rec['deflection_ks']:+.3e}  "
                f"freqKS={rec['vibration_ks']:+.3e}  buckKS={rec['buckling_ks']:+.3e}  "
                f"sl={rec['max_slenderness']:6.1f}  H={rec['H_m']:.3f}  c={rec['camber_m']:.3f}"
            , flush=True)

    constraints = {'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac}
    t0 = time.perf_counter()
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={'maxiter': int(maxiter), 'ftol': float(ftol), 'disp': bool(disp)},
    )
    wall = time.perf_counter() - t0

    af, vf = cache.ensure_analysis(res.x)
    cf = cons_fun(res.x)
    names = _constraint_names(nm, include_buckling)
    multipliers = getattr(res, 'multipliers', None)
    multiplier_map = None
    if multipliers is not None:
        # SLSQP returns equality multipliers first, then inequality multipliers.
        mult = np.asarray(multipliers, float).ravel()
        multiplier_map = {name: float(mult[i]) for i, name in enumerate(names[:len(mult)])}

    result = {
        'success': bool(res.success),
        'status': int(res.status),
        'message': str(res.message),
        'iterations': int(res.nit),
        'objective_scaled': float(res.fun),
        'wall_time_s': float(wall),
        'n_analyses': int(cache.n_analyses),
        'n_gradient_evals': int(cache.n_gradients),
        'analysis_time_s': float(cache.analysis_time),
        'gradient_time_s': float(cache.gradient_time),
        'include_buckling': bool(include_buckling),
        'initial': {
            'radii_m': r0.tolist(),
            'H_m': float(initial_H),
            'camber_m': float(initial_camber),
            **{k: float(v) for k,v in v0.items()},
            'max_slenderness': float((slenderness_values(a0).max()+1)*SLENDERNESS_LIMIT),
        },
        'final': {
            'radii_m': af.radii.tolist(),
            'H_m': float(af.H),
            'camber_m': float(af.camber),
            **{k: float(v) for k,v in vf.items()},
            'max_slenderness': float((slenderness_values(af).max()+1)*SLENDERNESS_LIMIT),
            'min_radius_m': float(np.min(af.radii)),
            'max_radius_m': float(np.max(af.radii)),
            'min_constraint_margin': float(np.min(cf)),
        },
        'constraint_names': names,
        'constraint_margins': [float(v) for v in cf],
        'multipliers': multiplier_map,
        'history': history,
    }
    return res, result, af


def write_optimization_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


@dataclass
class StaticEvaluationCache:
    nmembers: int
    x: np.ndarray | None = None
    analysis: object | None = None
    values: dict | None = None
    grads: dict | None = None
    n_analyses: int = 0
    n_gradients: int = 0

    def _same(self, x):
        return self.x is not None and np.array_equal(np.asarray(x), self.x)

    def ensure_analysis(self, x):
        from .sensitivity import analyze_static_design, static_response_values
        x=np.asarray(x,float)
        if not self._same(x):
            r,H,c=unpack_scaled_design(x,self.nmembers)
            self.analysis=analyze_static_design(r,H,c)
            self.values=static_response_values(self.analysis)
            self.grads=None; self.x=x.copy(); self.n_analyses+=1
        return self.analysis,self.values

    def ensure_gradients(self,x):
        from .sensitivity import reduced_static_gradients
        self.ensure_analysis(x)
        if self.grads is None:
            self.grads=reduced_static_gradients(self.analysis); self.n_gradients+=1
        return self.grads


def optimize_bridge_static_stage(
    initial_r=0.0755,
    initial_H=3.0,
    initial_camber=0.0,
    initial_x=None,
    maxiter=100,
    ftol=2e-7,
    disp=True,
):
    """Stage-1 optimization with R2, R3, and R6 only.

    R4/R5 are restored in the final full solve. This stage is purely a numerical accelerator:
    it moves the 179-variable design close to the stress/slenderness optimum using only the
    static PDE, so SLSQP line-search trials do not pay for eigenproblems.
    """
    base=build_footbridge(); nm=len(base.members); scales=design_scales(nm)
    if initial_x is None:
        r0=np.full(nm,float(initial_r)) if np.isscalar(initial_r) else np.asarray(initial_r,float)
        x0=pack_scaled_design(r0,initial_H,initial_camber)
    else:
        x0=np.asarray(initial_x,float).copy()
    cache=StaticEvaluationCache(nm)
    _a0,v0=cache.ensure_analysis(x0); mass_scale=float(v0['mass_total_kg'])
    history=[]

    def obj(x): return cache.ensure_analysis(x)[1]['mass_total_kg']/mass_scale
    def jac(x): return cache.ensure_gradients(x)['mass_total_kg']*scales/mass_scale
    def cfun(x):
        a,v=cache.ensure_analysis(x)
        return np.r_[-v['stress_ks'],-v['deflection_ks'],-slenderness_ks(a)[0]]
    def cjac(x):
        a,_=cache.ensure_analysis(x); g=cache.ensure_gradients(x)
        return np.vstack([-(g['stress_ks']*scales),-(g['deflection_ks']*scales),-slenderness_ks_jacobian_scaled(a)])
    def cb(x):
        a,v=cache.ensure_analysis(x); sl=(slenderness_values(a).max()+1)*SLENDERNESS_LIMIT
        rec={'iter':len(history),'mass':float(v['mass_total_kg']),'stress_ks':float(v['stress_ks']),
             'deflection_ks':float(v['deflection_ks']),'max_slenderness':float(sl),'H':float(a.H),
             'camber':float(a.camber),'min_r_mm':float(a.radii.min()*1e3),'max_r_mm':float(a.radii.max()*1e3)}
        history.append(rec)
        if disp:
            print(f"stage1 {rec['iter']:3d}: mass={rec['mass']:8.1f} stressKS={rec['stress_ks']:+.3e} deflKS={rec['deflection_ks']:+.3e} sl={sl:6.1f} H={a.H:.3f} c={a.camber:.3f}",flush=True)
    res=minimize(obj,x0,method='SLSQP',jac=jac,bounds=default_scaled_bounds(nm),
                 constraints={'type':'ineq','fun':cfun,'jac':cjac},callback=cb,
                 options={'maxiter':int(maxiter),'ftol':float(ftol),'disp':bool(disp)})
    return res,history,cache


def symmetric_member_groups(model=None, digits=8):
    """Group physical members by the bridge's geometric x/y reflection symmetries.

    The FE model still contains every physical member.  Only sizing variables are tied.
    A transformed member is grouped only when the transformed physical member actually exists,
    which keeps the single-diagonal top-bracing pattern from inventing nonexistent symmetry.
    """
    if model is None:
        model = build_footbridge()
    span = model.parameters['span']

    def pkey(p):
        return tuple(np.round(np.asarray(p,float), digits))
    def mkey(kind,a,b):
        aa,bb=pkey(a),pkey(b)
        return (kind,)+tuple(sorted((aa,bb)))

    lookup={mkey(m['kind'],m['a'],m['b']):i for i,m in enumerate(model.members)}
    parent=list(range(len(model.members)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj: parent[rj]=ri

    def transforms(p):
        x,y,z=np.asarray(p,float)
        return [
            np.array([x,y,z]),
            np.array([span-x,y,z]),
            np.array([x,-y,z]),
            np.array([span-x,-y,z]),
        ]

    for i,m in enumerate(model.members):
        ta=transforms(m['a']); tb=transforms(m['b'])
        for a,b in zip(ta,tb):
            j=lookup.get(mkey(m['kind'],a,b))
            if j is not None: union(i,j)

    buckets={}
    for i in range(len(model.members)):
        buckets.setdefault(find(i),[]).append(i)
    groups=sorted((sorted(v) for v in buckets.values()), key=lambda g:g[0])
    member_to_group=np.empty(len(model.members),dtype=int)
    for gi,g in enumerate(groups):
        member_to_group[g]=gi
    return groups,member_to_group


def group_radii_to_members(group_radii, groups):
    r=np.empty(sum(len(g) for g in groups),float)
    for gi,g in enumerate(groups): r[g]=group_radii[gi]
    return r


def member_gradient_to_groups(g_member, groups):
    g_member=np.asarray(g_member,float)
    return np.array([np.sum(g_member[g]) for g in groups],float)


def grouped_scales(ngroups):
    return np.r_[np.full(ngroups,0.05),2.0,0.5]


def pack_grouped_design(group_radii,H,camber):
    d=np.r_[np.asarray(group_radii,float),H,camber]
    return d/grouped_scales(len(group_radii))


def unpack_grouped_design(x,groups):
    ng=len(groups); d=np.asarray(x,float)*grouped_scales(ng)
    return group_radii_to_members(d[:ng],groups),float(d[ng]),float(d[ng+1]),d[:ng]


def grouped_bounds(ngroups):
    s=grouped_scales(ngroups)
    lo=np.r_[np.full(ngroups,.010),1.0,0.0]/s
    hi=np.r_[np.full(ngroups,.150),4.0,1.5]/s
    return list(zip(lo,hi))


@dataclass
class GroupedEvaluationCache:
    groups: list
    full: bool = True
    x: np.ndarray | None = None
    analysis: object | None = None
    values: dict | None = None
    grads: dict | None = None
    n_analyses: int = 0
    n_gradients: int = 0

    def _same(self,x): return self.x is not None and np.array_equal(np.asarray(x),self.x)
    def ensure_analysis(self,x):
        from .sensitivity import analyze_static_design, static_response_values
        x=np.asarray(x,float)
        if not self._same(x):
            r,H,c,_=unpack_grouped_design(x,self.groups)
            if self.full:
                self.analysis=analyze_design(r,H,c); self.values=response_values(self.analysis)
            else:
                self.analysis=analyze_static_design(r,H,c); self.values=static_response_values(self.analysis)
            self.x=x.copy(); self.grads=None; self.n_analyses+=1
        return self.analysis,self.values
    def ensure_gradients(self,x):
        from .sensitivity import reduced_static_gradients
        self.ensure_analysis(x)
        if self.grads is None:
            gp=reduced_gradients(self.analysis) if self.full else reduced_static_gradients(self.analysis)
            ng=len(self.groups); out={}
            for k,g in gp.items():
                out[k]=np.r_[member_gradient_to_groups(g[:len(self.analysis.radii)],self.groups),g[-2:]]
            self.grads=out; self.n_gradients+=1
        return self.grads


def grouped_slenderness_ks_jacobian_scaled(analysis,groups,rho=300.0):
    _g,w=slenderness_ks(analysis,rho=rho)
    Jp=slenderness_jacobian_scaled(analysis) # wrt full scaled x; convert to physical first
    # Jp member columns use full member scale=.05; shape columns scales 2,.5.
    ng=len(groups); Jg=np.zeros(ng+2)
    for gi,g in enumerate(groups): Jg[gi]=np.sum((w[:,None]*Jp)[:,g])
    Jg[-2]=w@Jp[:,-2]; Jg[-1]=w@Jp[:,-1]
    return Jg


def optimize_grouped_bridge(
    initial_group_r=0.0755,
    initial_H=3.0,
    initial_camber=0.0,
    initial_x=None,
    full=True,
    include_buckling=True,
    maxiter=100,
    ftol=2e-7,
    disp=True,
):
    """Optimize symmetry-tied member sizes plus H and camber.

    With ``full=False`` this is the static staging solve (R2,R3,R6). With ``full=True`` the
    final solve includes R4 and, optionally, R5. All constraints are checked on the full
    177-member FE model; only the sizing design variables are symmetry-tied.
    """
    base=build_footbridge(); groups,_=symmetric_member_groups(base); ng=len(groups)
    scales=grouped_scales(ng)
    if initial_x is None:
        gr=np.full(ng,float(initial_group_r)) if np.isscalar(initial_group_r) else np.asarray(initial_group_r,float)
        x0=pack_grouped_design(gr,initial_H,initial_camber)
    else:
        x0=np.asarray(initial_x,float).copy()
    cache=GroupedEvaluationCache(groups,full=full)
    a0,v0=cache.ensure_analysis(x0); mass_scale=float(v0['mass_total_kg']); history=[]
    pde=['stress_ks','deflection_ks']
    if full:
        pde.append('vibration_ks')
        if include_buckling: pde.append('buckling_ks')

    def obj(x): return cache.ensure_analysis(x)[1]['mass_total_kg']/mass_scale
    def jac(x): return cache.ensure_gradients(x)['mass_total_kg']*scales/mass_scale
    def cfun(x):
        a,v=cache.ensure_analysis(x)
        return np.r_[[-v[k] for k in pde], -slenderness_ks(a)[0]]
    def cjac(x):
        a,_=cache.ensure_analysis(x); g=cache.ensure_gradients(x)
        rows=[-(g[k]*scales) for k in pde]
        rows.append(-grouped_slenderness_ks_jacobian_scaled(a,groups))
        return np.vstack(rows)
    def cb(x):
        a,v=cache.ensure_analysis(x); sks=slenderness_ks(a)[0]
        rec={'iteration':len(history),'mass_total_kg':float(v['mass_total_kg']),
             'stress_ks':float(v['stress_ks']),'deflection_ks':float(v['deflection_ks']),
             'slenderness_ks':float(sks),'max_slenderness':float((slenderness_values(a).max()+1)*SLENDERNESS_LIMIT),
             'H_m':float(a.H),'camber_m':float(a.camber),'min_radius_mm':float(a.radii.min()*1e3),'max_radius_mm':float(a.radii.max()*1e3)}
        if full:
            rec.update(vibration_ks=float(v['vibration_ks']),buckling_ks=float(v['buckling_ks']),
                       lowest_frequency_hz=float(v['lowest_frequency_hz']),buckling_factor=float(v['buckling_factor']))
        history.append(rec)
        if disp:
            extra=(f" fKS={rec['vibration_ks']:+.2e} bKS={rec['buckling_ks']:+.2e}" if full else '')
            print(f"grouped {rec['iteration']:3d}: mass={rec['mass_total_kg']:8.1f} sKS={rec['stress_ks']:+.2e} dKS={rec['deflection_ks']:+.2e}{extra} slKS={rec['slenderness_ks']:+.2e} maxSl={rec['max_slenderness']:6.1f} H={a.H:.3f} c={a.camber:.3f}",flush=True)

    res=minimize(obj,x0,method='SLSQP',jac=jac,bounds=grouped_bounds(ng),
                 constraints={'type':'ineq','fun':cfun,'jac':cjac},callback=cb,
                 options={'maxiter':int(maxiter),'ftol':float(ftol),'disp':bool(disp)})
    af,vf=cache.ensure_analysis(res.x)
    return res,history,cache,af,vf,groups
