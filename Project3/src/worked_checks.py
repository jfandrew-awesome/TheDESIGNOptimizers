"""Write auditable numerical checks without using the training error routine."""
import csv
import io
import argparse
import json
import math
from pathlib import Path
import numpy as np
from nls_pinn import read_run_bytes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    root=Path(args.results)
    run='T4_seed0_soft_ic1_mass0'
    q=np.load(io.BytesIO(read_run_bytes(root/'predictions'/f'{run}.npz')))
    reference=np.load(root/'reference.npz')['q'][:len(q['t'])]
    pred=q['pred']
    # Scalar summation and sqrt, independent of np.linalg.norm / metrics().
    numerator=math.fsum((complex(a)-complex(b)).real**2+
                        (complex(a)-complex(b)).imag**2
                        for a,b in zip(pred.ravel(),reference.ravel()))
    denominator=math.fsum(complex(b).real**2+complex(b).imag**2 for b in reference.ravel())
    rel=math.sqrt(numerator/denominator)
    with open(root/'summary.csv',newline='') as f:
        logged=next(float(r['relative_l2']) for r in csv.DictReader(f) if r['run']==run)
    assert abs(rel-logged)<1e-12
    i,j=len(q['t'])-1,len(q['x'])//2
    p,r=complex(pred[i,j]),complex(reference[i,j])
    squared=(p.real-r.real)**2+(p.imag-r.imag)**2
    a=.2
    results=dict(
        plane_wave=dict(amplitude=1.2,wavenumber=1,omega=1.2**2-.5,
                        residual_coefficient=-(1.2**2-.5)-.5+1.2**2),
        analytic_initial_mass=2*math.pi*(1+a*a/2),
        analytic_initial_hamiltonian=2*math.pi*(a*a/4-.5*(1+3*a*a+3*a**4/8)),
        run=run,grid_point_count=int(pred.size),
        global_error_squared_sum=numerator,reference_squared_sum=denominator,
        scalar_computed_relative_l2=rel,logged_relative_l2=logged,
        pointwise=dict(t=float(q['t'][i]),x=float(q['x'][j]),
                       predicted_real=p.real,predicted_imag=p.imag,
                       reference_real=r.real,reference_imag=r.imag,
                       squared_absolute_error=squared,absolute_error=math.sqrt(squared)))
    (root/'worked_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
