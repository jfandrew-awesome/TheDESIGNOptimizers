"""Oracle initialization control for Path B; not a deployable data-free fix.

Same architecture, samples, physics loss and optimizer as the soft-IC PINN.
The only changed factor is initialization: use the supervised control's weights.
The costs of the supervised preparation and its reference data are retained.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv
import io
import argparse
import json
import time
from pathlib import Path
import numpy as np
from nls_pinn import (Network,Objective,train,predict,evaluate_jets,pde_loss,metrics,LENGTH,read_run_bytes)
from run_study import write_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    root=Path(args.results)
    with open(root/'summary.csv',newline='') as f:
        rows=list(csv.DictReader(f))
    if any(r['mode']=='oracle_warm_start' for r in rows):
        raise SystemExit('Warm-start controls already exist; use a fresh study directory.')
    config=json.loads((root/'config.json').read_text())
    data=np.load(root/'reference.npz')
    x,t,reference=data['x'],data['t'],data['q']
    for seed in config['seeds']:
        parent=next(r for r in rows if r['mode']=='supervised' and int(r['seed'])==seed)
        saved=np.load(io.BytesIO(read_run_bytes(root/'models'/f"{parent['run']}.npz")))
        net=Network(width=config['width'],seed=seed,end=4)
        net.theta0=saved['theta'].copy()
        xtrain=saved['x_train']
        indices=saved['time_indices']
        mode='oracle_warm_start'
        label=f'T4_seed{seed}_{mode}_ic1_mass0'
        print(label,flush=True)
        objective=Objective(net,xtrain,t[indices],mode,ic_weight=1,mass_weight=0)
        result,history,seconds=train(objective,config['iterations'])
        write_csv(root/'histories'/f'{label}.csv',history)
        np.savez_compressed(root/'models'/f'{label}.npz',theta=result.x,
                            x_train=xtrain,time_indices=indices)
        pred=predict(net,result.x,x,t,mode,anchor_nx=config['collocation_nx'])
        timing=[]
        for _ in range(5):
            start=time.perf_counter()
            predict(net,result.x,x[::2],t,mode,anchor_nx=config['collocation_nx'])
            timing.append(time.perf_counter()-start)
        inference=float(np.median(timing))
        numbers,series=metrics(pred,reference)
        heldout_x=LENGTH*(np.arange(64)+.5)/64
        heldout_t=t[:-1]+config['snapshot_dt']/2
        heldout_pde,_=pde_loss(evaluate_jets(net,result.x,heldout_x,heldout_t,
                                           mode,anchor_nx=config['collocation_nx']))
        _,_,parts=objective.evaluate(result.x)
        preparation=float(parent['train_seconds'])+float(parent['data_generation_seconds'])
        base=float(parent['baseline_seconds'])
        row=dict(run=label,horizon=4.,seed=seed,mode=mode,ic_weight=1.,mass_weight=0.,
                 iterations=int(result.nit),evaluations=int(result.nfev),
                 optimizer_success=bool(result.success),optimizer_message=str(result.message),
                 train_seconds=seconds,inference_seconds=inference,
                 data_generation_seconds=preparation,baseline_seconds=base,
                 formal_break_even_queries=(seconds+preparation)/(base-inference) if inference<base else None,
                 collocation_time_count=len(indices),heldout_pde_loss=heldout_pde,
                 **numbers,**{f'final_loss_{k}':v for k,v in parts.items()})
        rows.append(row)
        write_csv(root/'summary.csv',rows)
        np.savez_compressed(root/'predictions'/f'{label}.npz',x=x,t=t,pred=pred,**series)
        print(f"  L2={numbers['relative_l2']:.6g}, PDE={parts['pde']:.6g}, heldout={heldout_pde:.6g}",flush=True)
    (root/'warm_start_design.json').write_text(json.dumps(dict(
        mode='oracle_warm_start',purpose='Initialization control; not a practical remedy',
        changed_factor='Initialization from same-seed supervised fit',
        unchanged=['architecture','training samples','physics objective','optimizer','iteration budget'],
        cost_note='data_generation_seconds includes supervised training plus reference generation for this control.'),indent=2)+'\n')


if __name__=='__main__':
    main()
