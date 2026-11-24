"""V6: reduced-gradient checks for size + shape variables on the bridge."""
import pathlib, sys, time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.bridge import build_footbridge
from src.sensitivity import (analyze_design,response_values,reduced_gradients,
                             pack_scaled_design,unpack_scaled_design,design_scales)

KEYS=['mass_total_kg','stress_ks','deflection_ks','vibration_ks','buckling_ks']


def eval_x(x,nm):
    r,H,c=unpack_scaled_design(x,nm)
    return response_values(analyze_design(r,H,c))


def main():
    base_model=build_footbridge()
    radii=np.array([m['radius'] for m in base_model.members])
    nm=len(radii)
    x0=pack_scaled_design(radii,2.0,0.0)
    a0=analyze_design(radii,2.0,0.0)
    vals0=response_values(a0)
    t=time.time(); grads=reduced_gradients(a0); tg=time.time()-t

    rng=np.random.default_rng(34204)
    v=rng.normal(size=x0.size)
    # Deliberately include substantial shape content so H/camber are genuinely tested.
    v[-2:]=[2.0,-1.5]
    v/=np.linalg.norm(v)
    scales=design_scales(nm)
    directional={k: float((grads[k]*scales)@v) for k in KEYS}

    eps_list=np.array([1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4])
    errors={k:[] for k in KEYS}; cds={k:[] for k in KEYS}
    print('V6 bridge reduced-gradient directional check')
    print(f'  variables={x0.size} ({nm} radii + H + camber), gradient wall time={tg:.3f} s')
    for k in KEYS:
        print(f'  {k:18s}: value={vals0[k]: .9e}, adjoint dir={directional[k]: .9e}')
    print('  step sweep:')
    for eps in eps_list:
        vp=eval_x(x0+eps*v,nm); vm=eval_x(x0-eps*v,nm)
        line=[f'eps={eps:.1e}']
        for k in KEYS:
            cd=(vp[k]-vm[k])/(2*eps)
            err=abs(cd-directional[k])/max(1.0,abs(cd),abs(directional[k]))
            cds[k].append(cd); errors[k].append(err)
            line.append(f'{k.split("_")[0]} err={err:.2e}')
        print('    '+' | '.join(line))

    # At least one step should reach strong agreement for every response.
    limits={'mass_total_kg':2e-7,'stress_ks':3e-6,'deflection_ks':3e-6,
            'vibration_ks':1e-5,'buckling_ks':2e-5}
    for k in KEYS:
        best=min(errors[k])
        print(f'  best {k:18s} relative directional error = {best:.3e}')
        assert best < limits[k], (k,best,directional[k],cds[k])
    print('V6 directional gradients: PASS')

if __name__=='__main__': main()
