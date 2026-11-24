"""Build all assignment diagnostic figures from saved raw measurements."""
from pathlib import Path
import argparse
import csv
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nls_pinn import read_run_bytes

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','figure.dpi':140,
                     'savefig.dpi':180,'axes.grid':True,'grid.alpha':0.18})
COLORS=['#2457a6','#d35a35','#168478','#9a59ad','#b79428','#263549']


def rows(path):
    return list(csv.DictReader(io.StringIO(read_run_bytes(path).decode('utf-8'))))


def label(mode,ic=1,mass=0):
    if mode=='soft':
        return 'Soft IC' if float(ic)==1 and float(mass)==0 else (
            f'Soft IC, w={float(ic):g}' if float(ic)!=1 else 'Mass penalty, w=10')
    return {'hard_ic':'Exact IC','hard_mass':'Exact mass',
            'hard_ic_mass':'Exact IC + mass',
            'hard_ic_mass_dense':'Exact IC + mass, dense',
            'supervised':'Supervised control',
            'oracle_warm_start':'Oracle initialization'}[mode]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    root=Path(args.results)
    out=root/'figures'
    out.mkdir(exist_ok=True)
    data=rows(root/'summary.csv')
    reference=np.load(root/'reference.npz')
    def select(mode,seed=0,end=4,ic=1,mass=0):
        return next(r for r in data if r['mode']==mode and int(r['seed'])==seed
                    and float(r['horizon'])==end and float(r['ic_weight'])==ic
                    and float(r['mass_weight'])==mass)
    def prediction(r):
        p=dict(np.load(io.BytesIO(read_run_bytes(root/'predictions'/f"{r['run']}.npz"))))
        p['reference']=reference['q'][:len(p['t'])]
        return p
    def save(fig,name):
        fig.savefig(out/name,bbox_inches='tight',facecolor='white')
        plt.close(fig)

    # P2: all actual loss components, with zero/inactive terms labeled explicitly.
    modes=['soft','hard_ic','hard_mass','hard_ic_mass','hard_ic_mass_dense','supervised']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,mode in zip(axs.flat,modes):
        r=select(mode)
        history=rows(root/'histories'/f"{r['run']}.csv")
        it=np.array([float(h['iteration']) for h in history])
        for name,color in zip(['pde','initial','data','mass','total'],COLORS):
            y=np.array([float(h[name]) for h in history])
            if np.max(y)>1e-25:
                ax.semilogy(it,np.maximum(y,1e-25),label=name,color=color,lw=1.35)
        ax.set(title=label(mode),xlabel='L-BFGS iteration',ylabel='Unweighted loss')
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('P2 | Separate loss components (seed 0, T = 4)')
    save(fig,'loss_components.png')

    # P3: show every seed and use median + full observed range.
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for mode,color in zip(['soft','hard_ic'],COLORS):
        for ax,key,title in zip(axs,['relative_l2','final_relative_l2'],
                                ['Space-time relative L2','Final-time relative L2']):
            med,lo,hi=[],[],[]
            for end in [1,2,4]:
                rr=[r for r in data if r['mode']==mode and float(r['horizon'])==end
                    and float(r['ic_weight'])==1 and float(r['mass_weight'])==0]
                values=np.array([float(r[key]) for r in rr])
                med.append(np.median(values));lo.append(min(values));hi.append(max(values))
                ax.scatter(np.full(len(values),end),values,color=color,s=15,alpha=.6)
            ax.plot([1,2,4],med,'o-',color=color,label=label(mode))
            ax.fill_between([1,2,4],lo,hi,color=color,alpha=.12)
            ax.set(xlabel='Training horizon T (difficulty knob)',ylabel=title,yscale='log',xticks=[1,2,4])
            ax.legend(fontsize=8)
    fig.suptitle('P3 | Increasing the horizon exposes the failure')
    save(fig,'error_vs_horizon.png')

    # Controls at the hardest horizon; all three seeds are visible.
    specs=[('soft',1,0),('soft',10,0),('soft',100,0),('soft',1,10),
           ('hard_ic',1,0),('hard_mass',1,0),('hard_ic_mass',1,0),
           ('hard_ic_mass_dense',1,0),('supervised',1,0),('oracle_warm_start',1,0)]
    short=['Soft IC','IC w=10','IC w=100','Mass penalty','Exact IC','Exact mass',
           'Exact IC\n+ mass','Exact IC +\nmass, dense','Supervised','Oracle\ninitialization']
    fig,axs=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    for ax,key,title in zip(axs,['relative_l2','max_mass_deviation','max_hamiltonian_deviation'],
                            ['Space-time relative L2','Maximum relative mass defect',
                             'Maximum relative Hamiltonian defect']):
        for i,(mode,ic,mass) in enumerate(specs):
            vals=[float(select(mode,s,ic=ic,mass=mass)[key]) for s in [0,1,2]]
            ax.scatter([i-.1,i,i+.1],np.maximum(vals,1e-16),s=25,color=COLORS[i%len(COLORS)])
            ax.plot([i-.23,i+.23],[np.median(vals)]*2,color='#162c40',lw=2)
        ax.set(yscale='log',ylabel=title,xticks=range(len(short)),xticklabels=short)
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Controlled comparisons at T = 4 | three seeds; bars show medians')
    save(fig,'controlled_comparisons.png')

    # P4: common color scale for pointwise errors, and one separate reference panel.
    panel_modes=['soft','hard_mass','hard_ic_mass_dense','supervised','oracle_warm_start']
    predictions=[prediction(select(m)) for m in panel_modes]
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    ref=predictions[0]['reference']
    x,t=predictions[0]['x'],predictions[0]['t']
    im=axs[0,0].pcolormesh(x,t,np.abs(ref),shading='auto',cmap='viridis')
    axs[0,0].set(title='Fourier reference |q|',xlabel='x',ylabel='t')
    fig.colorbar(im,ax=axs[0,0],label='Magnitude')
    vmax=max(float(np.max(np.abs(p['pred']-p['reference']))) for p in predictions)
    for ax,mode,p in zip(list(axs.flat)[1:],panel_modes,predictions):
        im=ax.pcolormesh(p['x'],p['t'],np.abs(p['pred']-p['reference']),
                         shading='auto',vmin=0,vmax=vmax,cmap='magma')
        ax.set(title=label(mode)+' |error|',xlabel='x',ylabel='t')
    fig.colorbar(im,ax=list(axs.flat)[1:],label='Absolute complex field error',shrink=.8)
    fig.suptitle('P4 | Reference and pointwise errors (seed 0, T = 4)')
    save(fig,'error_fields.png')

    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    k=np.abs(np.rint(np.fft.fftfreq(len(x))*len(x))).astype(int)
    for mode,color,p in zip(panel_modes,COLORS,predictions):
        power=np.mean(np.abs(np.fft.fft(p['pred']-p['reference'],axis=1)/len(x))**2,axis=0)
        spectrum=np.array([power[k==m].sum() for m in range(65)])
        ax.semilogy(range(65),np.maximum(spectrum,1e-30),label=label(mode),color=color)
    power=np.mean(np.abs(np.fft.fft(ref,axis=1)/len(x))**2,axis=0)
    ax.semilogy(range(65),[max(power[k==m].sum(),1e-30) for m in range(65)],
                ':',color='#8a8a8a',label='Reference field power')
    ax.set(xlabel='Absolute Fourier wavenumber |k|',ylabel='Time-mean spectral power',
           title='P4 | Fourier spectrum of the error',ylim=(1e-22,10))
    ax.legend(fontsize=8,ncol=2)
    save(fig,'error_spectra.png')

    # Time series of phase and both invariants, beyond a single MSE value.
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    dx=2*np.pi/len(x)
    m0=dx*np.sum(np.abs(ref[0])**2)
    ksigned=np.fft.fftfreq(len(x),d=dx)*2*np.pi
    qx=np.fft.ifft(1j*ksigned*np.fft.fft(ref[0]))
    h0=dx*np.sum(.5*np.abs(qx)**2-.5*np.abs(ref[0])**4)
    for mode,color,p in zip(panel_modes,COLORS,predictions):
        axs[0,0].semilogy(p['t'],np.maximum(p['relative_l2'],1e-16),color=color,label=label(mode))
        axs[0,1].plot(p['t'],p['phase_offset'],color=color)
        axs[1,0].semilogy(p['t'],np.maximum(np.abs(p['mass']/m0-1),1e-16),color=color)
        axs[1,1].semilogy(p['t'],np.maximum(np.abs((p['hamiltonian']-h0)/h0),1e-16),color=color)
    for ax,title in zip(axs.flat,['Relative field error','Unwrapped overlap phase (radians)',
                                 'Relative mass defect','Relative Hamiltonian defect']):
        ax.set(title=title,xlabel='t')
    axs[0,0].legend(fontsize=7)
    axs[0,0].set_ylim(1e-4,3)
    axs[1,1].set_ylim(1e-6,1)
    fig.suptitle('Long-horizon diagnostics (seed 0) | phase is unreliable when overlap is small')
    save(fig,'long_time_diagnostics.png')

    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    for i,(mode,ic,mass) in enumerate(specs):
        for j,key in enumerate(['final_loss_pde','heldout_pde_loss']):
            vals=[float(select(mode,s,ic=ic,mass=mass)[key]) for s in [0,1,2]]
            ax.scatter(np.array([i-.06,i,i+.06])+(j-.5)*.25,vals,
                       marker='o' if j==0 else 'x',color=COLORS[j],s=26,
                       label=('Training residual' if j==0 else 'Held-out residual') if i==0 else None)
    ax.set(yscale='log',xticks=range(len(short)),xticklabels=short,ylabel='Mean squared PDE residual',
           title='Sampling check | shifted spatial points and intermediate times')
    ax.tick_params(axis='x',labelsize=8)
    ax.legend()
    save(fig,'heldout_residuals.png')

    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for mode,color in zip(['soft','oracle_warm_start'],COLORS):
        r=select(mode)
        h=rows(root/'histories'/f"{r['run']}.csv")
        axs[0].semilogy([float(a['iteration']) for a in h],
                        [float(a['total']) for a in h],color=color,label=label(mode))
        p=prediction(r)
        axs[1].semilogy(p['t'],np.maximum(p['relative_l2'],1e-16),color=color,label=label(mode))
    axs[0].set(title='Identical physics objective',xlabel='Physics-training iteration',ylabel='PDE loss + IC loss')
    axs[1].set(title='Solution error after physics training',xlabel='t',ylabel='Relative field error')
    for ax in axs:ax.legend(fontsize=8)
    fig.suptitle('Initialization control | same network, samples, objective and optimizer (seed 0)')
    save(fig,'initialization_control.png')
    print(f'Created {len(list(out.glob("*.png")))} diagnostic figures.')


if __name__=='__main__':
    main()
