"""Generate Milestone 4 gradient-check data and Taylor plot."""
import json, pathlib, sys, time
import numpy as np
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.bridge import build_footbridge
from src.sensitivity import (analyze_design,response_values,reduced_gradients,
    pack_scaled_design,unpack_scaled_design,design_scales)

KEYS=['mass_total_kg','stress_ks','deflection_ks','vibration_ks','buckling_ks']

base=build_footbridge(); radii=np.array([m['radius'] for m in base.members]); nm=len(radii)
x0=pack_scaled_design(radii,2.0,0.0)
t0=time.perf_counter(); a0=analyze_design(radii,2.0,0.0); analysis_s=time.perf_counter()-t0
vals=response_values(a0)
t0=time.perf_counter(); grads=reduced_gradients(a0); gradient_s=time.perf_counter()-t0
rng=np.random.default_rng(34204); v=rng.normal(size=x0.size); v[-2:]=[2.0,-1.5]; v/=np.linalg.norm(v)
scales=design_scales(nm)
dirder={k:float((grads[k]*scales)@v) for k in KEYS}
eps=np.array([1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4])
errors={k:[] for k in KEYS}; central={k:[] for k in KEYS}

def ev(x):
    r,H,c=unpack_scaled_design(x,nm)
    return response_values(analyze_design(r,H,c))

for h in eps:
    vp=ev(x0+h*v); vm=ev(x0-h*v)
    for k in KEYS:
        cd=(vp[k]-vm[k])/(2*h)
        central[k].append(float(cd))
        errors[k].append(float(abs(cd-dirder[k])/max(1.0,abs(cd),abs(dirder[k]))))

fig,ax=plt.subplots(figsize=(7.2,4.8))
for k in KEYS:
    ax.loglog(eps,errors[k],marker='o',label=k.replace('_',' '))
ref=errors['stress_ks'][0]*(eps/eps[0])**2
ax.loglog(eps,ref,linestyle='--',label='second-order reference')
ax.set_xlabel('central-difference step in scaled design')
ax.set_ylabel('relative directional-derivative error')
ax.set_title('V6 Taylor test: adjoint/eigen sensitivities vs. central differences')
ax.grid(True,which='both',alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/'assets'/'gradient_taylor.png',dpi=180); plt.close(fig)

fd_est=2*x0.size*analysis_s
result={
 'n_variables':int(x0.size),'n_member_radii':nm,'analysis_seconds':analysis_s,
 'all_gradients_seconds':gradient_s,'central_fd_jacobian_est_seconds':fd_est,
 'estimated_speedup_vs_full_central_fd':fd_est/gradient_s,
 'responses':vals,'directional_derivatives':dirder,'steps':eps.tolist(),
 'relative_errors':errors,'central_difference_directionals':central,
 'best_relative_error':{k:min(errors[k]) for k in KEYS},
 'note':'Shape partial derivatives use centered geometry differentiation at fixed state; state dependence is handled by the adjoint. No shape-perturbation PDE solves occur inside reduced_gradients.'
}
(ROOT/'assets'/'milestone4_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['n_variables','analysis_seconds','all_gradients_seconds','central_fd_jacobian_est_seconds','estimated_speedup_vs_full_central_fd','best_relative_error']},indent=2))
