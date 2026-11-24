"""Independent checks for signs, Fourier reference, input jets and gradients."""
import json
import numpy as np
from nls_pinn import (Network, Objective, initial, spectral_solution,
                      model_forward, predict, evaluate_jets, invariants, LENGTH)


def verify():
    results = {}
    x, t, q, _ = spectral_solution(nx=64, dt=0.001, end=1, save_dt=0.025,
                                  plane_wave=True)
    exact = 1.2*np.exp(1j*(x[None, :]+(1.2**2-0.5)*t[:, None]))
    results['plane_wave_max_abs_error'] = float(np.max(np.abs(q-exact)))
    assert results['plane_wave_max_abs_error'] < 1e-10
    mass, ham = invariants(q)
    assert np.max(np.abs(mass-LENGTH*1.2**2)) < 1e-10
    assert np.max(np.abs(ham-LENGTH*(0.5*1.2**2-0.5*1.2**4))) < 1e-10

    rng = np.random.default_rng(27)
    net = Network(width=6, seed=3, end=2)
    xx, tt = rng.uniform(0, LENGTH, 11), rng.uniform(0, 2, 11)
    jets, _ = net.forward(net.theta0, xx, tt)
    eps = 1e-4
    values = lambda a,b: net.values(net.theta0,a,b)
    fd_t = (values(xx,tt+eps)-values(xx,tt-eps))/(2*eps)
    fd_x = (values(xx+eps,tt)-values(xx-eps,tt))/(2*eps)
    fd_xx = (values(xx+eps,tt)-2*values(xx,tt)+values(xx-eps,tt))/eps**2
    results['input_derivative_max_errors'] = [float(np.max(np.abs(fd-j)))
                                              for fd,j in zip([fd_t,fd_x,fd_xx],jets[1:])]
    assert max(results['input_derivative_max_errors']) < 1e-5
    xg = LENGTH*np.arange(12)/12
    times = np.array([0.2,0.7,1.4])
    results['directional_gradient_scaled_errors'] = {}
    for mode in ['soft','hard_ic','hard_mass','hard_ic_mass','supervised']:
        targets = rng.normal(size=(len(xg)*len(times), 2)) if mode=='supervised' else None
        obj = Objective(net,xg,times,mode,ic_weight=3,mass_weight=0.7,targets=targets)
        _,g,_ = obj.evaluate(net.theta0)
        errors=[]
        for _ in range(5):
            d = rng.normal(size=g.size)
            d /= np.linalg.norm(d)
            h = 1e-6
            fd = (obj.evaluate(net.theta0+h*d)[0]-obj.evaluate(net.theta0-h*d)[0])/(2*h)
            exact_g = float(g@d)
            errors.append(abs(fd-exact_g)/max(1,abs(fd),abs(exact_g)))
        results['directional_gradient_scaled_errors'][mode] = max(errors)
        assert max(errors)<1e-6,(mode,errors)
    for mode in ['hard_ic','hard_ic_mass']:
        value = predict(net,net.theta0,xg,np.array([0.0]),mode,anchor_nx=12)
        assert np.max(np.abs(value[0]-initial(xg)[:,0]))<1e-12
    value = predict(net,net.theta0,xg,times,'hard_mass',anchor_nx=12)
    mass,_=invariants(value)
    results['hard_mass_max_relative_defect_on_anchor_grid'] = float(np.max(np.abs(mass/(LENGTH*1.02)-1)))
    assert results['hard_mass_max_relative_defect_on_anchor_grid']<1e-12
    # Check the derivatives of the actual constrained field on an independent grid.
    xtest=LENGTH*(np.arange(17)+0.3)/17
    ttest=np.array([0.25,0.81,1.65])
    jj=evaluate_jets(net,net.theta0,xtest,ttest,'hard_ic_mass',anchor_nx=12)
    as_complex=lambda a: (a[:,0]+1j*a[:,1]).reshape(len(ttest),len(xtest))
    p=lambda x,t: predict(net,net.theta0,x,t,'hard_ic_mass',anchor_nx=12)
    h=1e-4
    fdt=(p(xtest,ttest+h)-p(xtest,ttest-h))/(2*h)
    fdxx=(p(xtest+h,ttest)-2*p(xtest,ttest)+p(xtest-h,ttest))/h**2
    results['constrained_field_derivative_max_errors']=[
        float(np.max(np.abs(fdt-as_complex(jj[1])))),
        float(np.max(np.abs(fdxx-as_complex(jj[3]))))]
    assert max(results['constrained_field_derivative_max_errors'])<2e-5
    return results


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
