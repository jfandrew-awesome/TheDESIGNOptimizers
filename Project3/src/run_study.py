"""Run the controlled NLS study and preserve every numerical result.

Examples:
  python src/run_study.py --pilot
  python src/run_study.py
  python src/run_study.py --out results-new --iterations 2000
"""
import os
# Keep the small matrix multiplications reproducible and avoid oversubscription.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
import matplotlib
from nls_pinn import (Network, Objective, spectral_solution, train, predict,
                      evaluate_jets, pde_loss, invariants, metrics, LENGTH)
from verify import verify


def write_csv(path, rows):
    if not rows:
        return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--iterations',type=int,default=1500)
    parser.add_argument('--out',default='results')
    args=parser.parse_args()
    out=Path(args.out)
    if (out/'summary.csv').exists():
        raise SystemExit('Output already contains a study. Use a new --out directory.')
    for sub in ['histories','predictions','models','figures']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    checks=verify()
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    horizons=[1.0,2.0,4.0]
    seeds=[0] if args.pilot else [0,1,2]
    config=dict(horizons=horizons,seeds=seeds,width=24,iterations=args.iterations,
                collocation_nx=32,time_samples_per_unit=12,
                reference_nx=256,reference_dt=0.0005,snapshot_dt=0.025,
                ic_amplitude=0.2,ic_weights=[1.0,10.0,100.0],
                notes='Uniform x grid; fixed seeded time subsets; same init within each seed and horizon.')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    environment=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
                     machine=platform.machine(),logical_cpu_count=os.cpu_count(),
                     numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                     device='CPU; no GPU',linear_algebra_threads=1,
                     timing_note='Wall clock in a shared hosted environment; not a dedicated benchmark.')
    try:
        environment['cpu_model']=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
    except (OSError,StopIteration):
        environment['cpu_model']='not exposed by execution environment'
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')

    print('Computing Fourier reference and convergence controls...',flush=True)
    x,t,reference,reference_sec=spectral_solution()
    np.savez_compressed(out/'reference.npz',x=x,t=t,q=reference)
    controls=[]
    for label,nx,dt in [('production',128,0.001),('time_coarse',256,0.001),
                        ('time_finer',256,0.00025),('space_coarse',128,0.0005),
                        ('space_finer',512,0.0005)]:
        _,_,q,seconds=spectral_solution(nx=nx,dt=dt)
        if nx<256:
            r=reference[:,::(256//nx)]
        else:
            r=reference
            q=q[:,::(nx//256)]
        numbers,_=metrics(q,r)
        controls.append(dict(label=label,nx=nx,dt=dt,wall_seconds=seconds,**numbers))
    write_csv(out/'baseline_convergence.csv',controls)
    reference_stats,_=metrics(reference,reference)
    (out/'baseline.json').write_text(json.dumps(dict(reference_seconds=reference_sec,
        reference_invariant_deviation=reference_stats,convergence=controls),indent=2)+'\n')
    # Benchmark the conventional solver on each actual output query.
    baseline_times={}
    for end in horizons:
        timings=[spectral_solution(nx=128,dt=0.001,end=end)[3] for _ in range(3)]
        baseline_times[str(end)]=dict(median_seconds=float(np.median(timings)),samples_seconds=timings)
    (out/'baseline_query_times.json').write_text(json.dumps(baseline_times,indent=2)+'\n')

    experiments=[]
    for end in horizons:
        for seed in seeds:
            experiments.extend([(end,seed,'soft',1.0,0.0),
                                (end,seed,'hard_ic',1.0,0.0)])
    # Diagnostic controls and extra structure variants on the hardest horizon.
    for seed in seeds:
        experiments.extend([(4.0,seed,'soft',10.0,0.0),
                            (4.0,seed,'soft',100.0,0.0),
                            (4.0,seed,'soft',1.0,10.0),
                            (4.0,seed,'hard_mass',1.0,0.0),
                            (4.0,seed,'hard_ic_mass',1.0,0.0),
                            (4.0,seed,'hard_ic_mass_dense',1.0,0.0),
                            (4.0,seed,'supervised',1.0,0.0)])
    if args.pilot:
        # A pilot is a separate, explicitly labeled run, never pooled with the study.
        experiments=[e for e in experiments if e[0] in (1.0,4.0)]
    summaries=[]
    for run,(end,seed,mode,ic_weight,mass_weight) in enumerate(experiments,1):
        label=f'T{end:g}_seed{seed}_{mode}_ic{ic_weight:g}_mass{mass_weight:g}'
        print(f'[{run}/{len(experiments)}] {label}',flush=True)
        net=Network(width=config['width'],seed=seed,end=end)
        end_index=int(round(end/config['snapshot_dt']))
        te=t[:end_index+1]
        ref=reference[:end_index+1]
        rng=np.random.default_rng(1000+seed)
        indices=np.sort(rng.choice(np.arange(1,end_index+1),
                                   size=int(config['time_samples_per_unit']*end),replace=False))
        if mode.endswith('_dense'):
            indices=np.arange(1,end_index+1)
        xtrain=LENGTH*np.arange(config['collocation_nx'])/config['collocation_nx']
        target_complex=reference[indices,::(config['reference_nx']//config['collocation_nx'])]
        targets=np.stack((target_complex.real,target_complex.imag),axis=2).reshape(-1,2)
        objective=Objective(net,xtrain,t[indices],mode,ic_weight,mass_weight,
                            targets=targets if mode=='supervised' else None)
        result,history,seconds=train(objective,args.iterations)
        write_csv(out/'histories'/f'{label}.csv',history)
        np.savez_compressed(out/'models'/f'{label}.npz',theta=result.x,
                            time_indices=indices,x_train=xtrain)
        pred=predict(net,result.x,x,te,mode,anchor_nx=config['collocation_nx'])
        timings=[]
        for _ in range(5):
            start=time.perf_counter()
            predict(net,result.x,x[::2],te,mode,anchor_nx=config['collocation_nx'])
            timings.append(time.perf_counter()-start)
        inference=float(np.median(timings))
        numbers,series=metrics(pred,ref)
        heldout_x=LENGTH*(np.arange(64)+0.5)/64
        heldout_t=te[:-1]+config['snapshot_dt']/2
        heldout_pde,_=pde_loss(evaluate_jets(net,result.x,heldout_x,heldout_t,
                                           mode,anchor_nx=config['collocation_nx']))
        # The cost comparison uses equal 128-point output grids and full trajectories.
        base=baseline_times[str(end)]['median_seconds']
        data_generation=reference_sec if mode=='supervised' else 0.0
        nstar=(seconds+data_generation)/(base-inference) if inference<base else None
        _,_,parts=objective.evaluate(result.x)
        row=dict(run=label,horizon=end,seed=seed,mode=mode,ic_weight=ic_weight,
                 mass_weight=mass_weight,iterations=int(result.nit),evaluations=int(result.nfev),
                 optimizer_success=bool(result.success),optimizer_message=str(result.message),
                 train_seconds=seconds,inference_seconds=inference,
                 data_generation_seconds=data_generation,
                 baseline_seconds=base,formal_break_even_queries=nstar,
                 collocation_time_count=len(indices),heldout_pde_loss=heldout_pde,
                 **numbers,**{f'final_loss_{k}':v for k,v in parts.items()})
        summaries.append(row)
        write_csv(out/'summary.csv',summaries)
        np.savez_compressed(out/'predictions'/f'{label}.npz',x=x,t=te,pred=pred,
                            reference=ref,**series)
        print(f"  relative L2={numbers['relative_l2']:.5g}; final={numbers['final_relative_l2']:.5g}; "
              f"mass defect={numbers['max_mass_deviation']:.5g}; training={seconds:.2f}s",flush=True)
    print(f'Done. Raw results: {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
