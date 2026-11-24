"""Small, double-precision periodic NLS PINN implemented with NumPy.

The forward pass propagates value, t derivative, x derivative and xx derivative.
Reverse mode differentiates through all four quantities. No finite-difference
derivatives are used for training. See verify.py for independent numerical checks.
"""
from __future__ import annotations

import time
from pathlib import Path
import zipfile
import numpy as np
from scipy.optimize import minimize

LENGTH = 2.0 * np.pi
AMPLITUDE = 0.2


def read_run_bytes(path):
    """Read an ordinary result file or its entry in a compact seed archive.

    Archives are read in memory, so plotting never expands the upload folder
    beyond the browser's file-count limit. Fresh study runs still use files.
    """
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    if path.parent.name not in {'histories', 'models', 'predictions'}:
        raise FileNotFoundError(path)
    member = f'{path.parent.name}/{path.name}'
    for archive in sorted(path.parent.parent.glob('runs-seed-*.zip')):
        with zipfile.ZipFile(archive) as saved:
            try:
                return saved.read(member)
            except KeyError:
                pass
    raise FileNotFoundError(path)


def initial(x):
    x = np.asarray(x)
    return np.column_stack((1.0 + AMPLITUDE * np.cos(x), np.zeros(x.size)))


def initial_jets(x):
    z = np.zeros((len(x), 2))
    q = initial(x)
    qx = z.copy()
    qxx = z.copy()
    qx[:, 0] = -AMPLITUDE * np.sin(x)
    qxx[:, 0] = -AMPLITUDE * np.cos(x)
    return [q, z, qx, qxx]


def spectral_solution(nx=256, dt=0.0005, end=4.0, save_dt=0.025,
                      plane_wave=False):
    """Strang nonlinear/linear/nonlinear Fourier splitting; periodic grid."""
    start = time.perf_counter()
    x = LENGTH * np.arange(nx) / nx
    steps = int(round(end / dt))
    stride = int(round(save_dt / dt))
    if not np.isclose(steps * dt, end) or not np.isclose(stride * dt, save_dt):
        raise ValueError('end and save_dt must be integer multiples of dt')
    if steps % stride:
        raise ValueError('end must be an integer multiple of save_dt')
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=LENGTH / nx)
    linear = np.exp(-0.5j * k * k * dt)
    q = (1.2 * np.exp(1j * x) if plane_wave
         else initial(x)[:, 0].astype(complex))
    states = [q.copy()]
    for step in range(1, steps + 1):
        q *= np.exp(0.5j * dt * np.abs(q) ** 2)
        q = np.fft.ifft(linear * np.fft.fft(q))
        q *= np.exp(0.5j * dt * np.abs(q) ** 2)
        if step % stride == 0:
            states.append(q.copy())
    return x, np.arange(len(states)) * save_dt, np.asarray(states), time.perf_counter() - start


class Network:
    """tanh MLP on (cos x, sin x, cos 2x, sin 2x, 2t/T-1)."""

    def __init__(self, width=24, seed=0, end=1.0):
        self.width, self.end = width, float(end)
        self.sizes = [5, width, width, 2]
        rng = np.random.default_rng(seed)
        parts = []
        for ni, no in zip(self.sizes[:-1], self.sizes[1:]):
            parts.extend([rng.normal(0, np.sqrt(2 / (ni + no)), (ni, no)).ravel(),
                          np.zeros(no)])
        self.theta0 = np.concatenate(parts)

    def unpack(self, theta):
        layers, offset = [], 0
        for ni, no in zip(self.sizes[:-1], self.sizes[1:]):
            w = theta[offset:offset + ni * no].reshape(ni, no)
            offset += ni * no
            b = theta[offset:offset + no]
            offset += no
            layers.append((w, b))
        return layers

    def inputs(self, x, t):
        h = np.column_stack((np.cos(x), np.sin(x), np.cos(2*x),
                             np.sin(2*x), 2*t/self.end - 1))
        ht = np.zeros_like(h)
        ht[:, -1] = 2 / self.end
        hx = np.column_stack((-np.sin(x), np.cos(x), -2*np.sin(2*x),
                              2*np.cos(2*x), np.zeros_like(x)))
        hxx = np.column_stack((-np.cos(x), -np.sin(x), -4*np.cos(2*x),
                               -4*np.sin(2*x), np.zeros_like(x)))
        return [h, ht, hx, hxx]

    def forward(self, theta, x, t):
        jets = self.inputs(x, t)
        cache = []
        layers = self.unpack(theta)
        for i, (w, b) in enumerate(layers):
            z = [jets[0] @ w + b] + [h @ w for h in jets[1:]]
            hidden = i < len(layers) - 1
            cache.append((jets, z, w, hidden))
            if hidden:
                f = np.tanh(z[0])
                fp = 1 - f*f
                fpp = -2*f*fp
                jets = [f, fp*z[1], fp*z[2], fpp*z[2]**2 + fp*z[3]]
            else:
                jets = z
        return jets, cache

    def backward(self, cache, adjoints):
        gradients = []
        g = adjoints
        for h, z, w, hidden in reversed(cache):
            if hidden:
                f = np.tanh(z[0])
                fp = 1-f*f
                fpp = -2*f*fp
                fppp = (6*f*f-2)*fp
                gz = [g[0]*fp + g[1]*fpp*z[1] + g[2]*fpp*z[2]
                      + g[3]*(fppp*z[2]**2 + fpp*z[3]),
                      g[1]*fp,
                      g[2]*fp + 2*g[3]*fpp*z[2],
                      g[3]*fp]
            else:
                gz = g
            gw = sum(a.T @ b for a, b in zip(h, gz))
            gb = gz[0].sum(axis=0)
            gradients.append((gw, gb))
            g = [a @ w.T for a in gz]
        return np.concatenate([a.ravel() for pair in reversed(gradients) for a in pair])

    def values(self, theta, x, t):
        h = np.column_stack((np.cos(x), np.sin(x), np.cos(2*x),
                             np.sin(2*x), 2*t/self.end - 1))
        layers = self.unpack(theta)
        for i, (w, b) in enumerate(layers):
            h = h @ w + b
            if i < len(layers)-1:
                h = np.tanh(h)
        return h


def raw_model(net, theta, x, t, mode):
    jets, cache = net.forward(theta, x, t)
    hard_ic = 'hard_ic' in mode
    if hard_ic:
        s = (t/net.end)[:, None]
        q0 = initial_jets(x)
        raw = [q0[0] + s*jets[0], jets[0]/net.end + s*jets[1],
               q0[2] + s*jets[2], q0[3] + s*jets[3]]
    else:
        raw = jets
    return raw, (cache, t, hard_ic)


def raw_backward(net, context, g):
    cache, t, hard_ic = context
    if hard_ic:
        s = (t/net.end)[:, None]
        g = [s*g[0] + g[1]/net.end, s*g[1], s*g[2], s*g[3]]
    return net.backward(cache, g)


def mass_normalize(raw, nx):
    """Enforce the quadrature mass exactly, retaining its t derivative."""
    r = [a.reshape(-1, nx, 2) for a in raw]
    mu0 = 1 + AMPLITUDE**2/2
    s = np.mean(np.sum(r[0]**2, axis=2), axis=1)
    if np.any(s < 1e-24):
        raise FloatingPointError('Mass-normalization denominator is zero')
    d = np.mean(np.sum(r[0]*r[1], axis=2), axis=1)
    c = np.sqrt(mu0/s)
    ct = -c*d/s
    cc, cct = c[:, None, None], ct[:, None, None]
    q = [cc*r[0], cc*r[1]+cct*r[0], cc*r[2], cc*r[3]]
    return [a.reshape(-1, 2) for a in q], (r, s, d, c, ct, nx)


def mass_backward(context, adjoints):
    r, s, d, c, ct, nx = context
    g = [a.reshape(-1, nx, 2) for a in adjoints]
    cc, cct = c[:, None, None], ct[:, None, None]
    result = [cc*g[0]+cct*g[1], cc*g[1], cc*g[2], cc*g[3]]
    gc = sum(np.sum(a*b, axis=(1, 2)) for a, b in zip(g, r))
    gct = np.sum(g[1]*r[0], axis=(1, 2))
    gc -= gct*d/s
    gd = -gct*c/s
    gs = gct*c*d/s**2 - 0.5*gc*c/s
    result[0] += (2*gs[:, None, None]*r[0] + gd[:, None, None]*r[1])/nx
    result[1] += gd[:, None, None]*r[0]/nx
    return [a.reshape(-1, 2) for a in result]


def model_forward(net, theta, x, t, mode, nx):
    raw, raw_context = raw_model(net, theta, x, t, mode)
    if 'mass' in mode:
        q, mass_context = mass_normalize(raw, nx)
    else:
        q, mass_context = raw, None
    return q, (raw_context, mass_context)


def model_backward(net, context, g):
    raw_context, mass_context = context
    if mass_context is not None:
        g = mass_backward(mass_context, g)
    return raw_backward(net, raw_context, g)


def pde_loss(jets):
    q, qt, _, qxx = jets
    a, b = q.T
    rho = a*a+b*b
    fr = -qt[:, 1] + 0.5*qxx[:, 0] + rho*a
    fi = qt[:, 0] + 0.5*qxx[:, 1] + rho*b
    n = len(q)
    g = [np.zeros_like(q) for _ in range(4)]
    g[0][:, 0] = (2*fr*(3*a*a+b*b) + 4*fi*a*b)/n
    g[0][:, 1] = (4*fr*a*b + 2*fi*(a*a+3*b*b))/n
    g[1][:, 0] = 2*fi/n
    g[1][:, 1] = -2*fr/n
    g[3][:, 0] = fr/n
    g[3][:, 1] = fi/n
    return float(np.mean(fr*fr+fi*fi)), g


class Objective:
    def __init__(self, net, xgrid, times, mode='soft', ic_weight=1.0,
                 mass_weight=0.0, targets=None):
        self.net, self.nx, self.mode = net, len(xgrid), mode
        self.x = np.tile(xgrid, len(times))
        self.t = np.repeat(times, len(xgrid))
        self.x0, self.t0 = xgrid, np.zeros(len(xgrid))
        self.ic_weight, self.mass_weight = ic_weight, mass_weight
        self.targets = targets

    def evaluate(self, theta):
        q, ctx = model_forward(self.net, theta, self.x, self.t, self.mode, self.nx)
        lpde, g = pde_loss(q)
        qi, ictx = model_forward(self.net, theta, self.x0, self.t0, self.mode, self.nx)
        delta0 = qi[0] - initial(self.x0)
        lic = float(np.mean(np.sum(delta0**2, axis=1)))
        gi = [np.zeros_like(qi[0]) for _ in range(4)]
        gi[0] = 2*delta0/self.nx
        means = np.mean(np.sum(q[0].reshape(-1, self.nx, 2)**2, axis=2), axis=1)
        mu0 = 1+AMPLITUDE**2/2
        defects = means/mu0 - 1
        lmass = float(np.mean(defects**2))
        ldata = 0.0
        if self.mode == 'supervised':
            delta = q[0] - self.targets
            ldata = float(np.mean(np.sum(delta**2, axis=1)))
            g = [np.zeros_like(q[0]) for _ in range(4)]
            g[0] = 2*delta/len(delta)
            loss = ldata + self.ic_weight*lic
        else:
            loss = lpde + self.ic_weight*lic + self.mass_weight*lmass
            g[0] += (4*self.mass_weight/(len(means)*self.nx*mu0)
                     * np.repeat(defects, self.nx)[:, None] * q[0])
        grad = model_backward(self.net, ctx, g)
        grad += self.ic_weight*model_backward(self.net, ictx, gi)
        parts = dict(pde=lpde, initial=lic, boundary=0.0, data=ldata,
                     mass=lmass, total=float(loss))
        return float(loss), grad, parts


def train(objective, iterations=1500):
    logs = []
    started = time.perf_counter()
    last = {}

    def fun(theta):
        loss, grad, parts = objective.evaluate(theta)
        last.update(parts)
        return loss, grad

    def callback(theta):
        logs.append(dict(iteration=len(logs), elapsed_seconds=time.perf_counter()-started,
                         **last))

    _, _, first = objective.evaluate(objective.net.theta0)
    logs.append(dict(iteration=0, elapsed_seconds=0.0, **first))
    result = minimize(fun, objective.net.theta0.copy(), method='L-BFGS-B', jac=True,
                      callback=callback,
                      options=dict(maxiter=iterations, maxfun=iterations*3,
                                   ftol=1e-13, gtol=1e-8, maxcor=30, maxls=40))
    elapsed = time.perf_counter()-started
    return result, logs, elapsed


def predict(net, theta, xgrid, times, mode='soft', anchor_nx=32):
    """Value-only inference. Mass normalization ALWAYS uses its training grid.

    Evaluating on a finer grid therefore independently checks quadrature error.
    """
    x = np.tile(xgrid, len(times))
    t = np.repeat(times, len(xgrid))
    q = net.values(theta, x, t)
    if 'hard_ic' in mode:
        q = initial(x) + (t/net.end)[:, None]*q
    if 'mass' in mode:
        xa = LENGTH*np.arange(anchor_nx)/anchor_nx
        xx, tt = np.tile(xa, len(times)), np.repeat(times, anchor_nx)
        anchors = net.values(theta, xx, tt)
        if 'hard_ic' in mode:
            anchors = initial(xx)+(tt/net.end)[:, None]*anchors
        means = np.mean(np.sum(anchors.reshape(-1, anchor_nx, 2)**2, axis=2), axis=1)
        c = np.sqrt((1+AMPLITUDE**2/2)/means)
        q = q.reshape(-1, len(xgrid), 2)*c[:, None, None]
    q = q.reshape(-1, len(xgrid), 2)
    return q[:, :, 0]+1j*q[:, :, 1]


def evaluate_jets(net, theta, xgrid, times, mode='soft', anchor_nx=32):
    """Derivatives on held-out points, with the original normalization anchors."""
    xx,tt = np.tile(xgrid,len(times)),np.repeat(times,len(xgrid))
    raw,_ = raw_model(net,theta,xx,tt,mode)
    if 'mass' not in mode:
        return raw
    xa=LENGTH*np.arange(anchor_nx)/anchor_nx
    anchors,_=raw_model(net,theta,np.tile(xa,len(times)),
                       np.repeat(times,anchor_nx),mode)
    _,context=mass_normalize(anchors,anchor_nx)
    c,ct=context[3],context[4]
    r=[a.reshape(len(times),len(xgrid),2) for a in raw]
    cc,cct=c[:,None,None],ct[:,None,None]
    return [a.reshape(-1,2) for a in
            [cc*r[0],cc*r[1]+cct*r[0],cc*r[2],cc*r[3]]]


def invariants(q):
    nx = q.shape[1]
    k = 2*np.pi*np.fft.fftfreq(nx, d=LENGTH/nx)
    qx = np.fft.ifft(1j*k[None, :]*np.fft.fft(q, axis=1), axis=1)
    mass = LENGTH*np.mean(np.abs(q)**2, axis=1)
    hamiltonian = LENGTH*np.mean(0.5*np.abs(qx)**2-0.5*np.abs(q)**4, axis=1)
    return mass, hamiltonian


def metrics(pred, reference):
    mass, ham = invariants(pred)
    mref, href = invariants(reference)
    err = pred-reference
    relative_t = np.linalg.norm(err, axis=1)/np.linalg.norm(reference, axis=1)
    overlap = np.sum(np.conj(reference)*pred, axis=1)
    phase = np.unwrap(np.angle(overlap))
    aligned = pred*np.exp(-1j*phase[:, None])
    aligned_error = np.linalg.norm(aligned-reference, axis=1)/np.linalg.norm(reference, axis=1)
    correlation = np.abs(overlap)/(np.linalg.norm(reference, axis=1)*np.linalg.norm(pred, axis=1)+1e-30)
    numbers = dict(
        relative_l2=float(np.linalg.norm(err)/np.linalg.norm(reference)),
        final_relative_l2=float(relative_t[-1]),
        final_phase_offset=float(phase[-1]),
        final_correlation=float(correlation[-1]),
        final_phase_aligned_l2=float(aligned_error[-1]),
        max_mass_deviation=float(np.max(np.abs(mass-mref[0]))/mref[0]),
        max_mass_drift=float(np.max(np.abs(mass-mass[0]))/mref[0]),
        max_hamiltonian_deviation=float(np.max(np.abs(ham-href[0]))/abs(href[0])),
        max_hamiltonian_drift=float(np.max(np.abs(ham-ham[0]))/abs(href[0])),
    )
    series = dict(relative_l2=relative_t, phase_offset=phase,
                  phase_aligned_l2=aligned_error, correlation=correlation,
                  mass=mass, hamiltonian=ham)
    return numbers, series
