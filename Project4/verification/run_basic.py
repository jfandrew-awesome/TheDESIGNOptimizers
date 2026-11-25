import sys, pathlib, numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from src.sections import chs_properties
from src.assembly import assemble, solve_static


def rel(a,b): return abs(a-b)/max(abs(b),1e-30)

def v1_cantilever():
    E=200e9; nu=.3; rho=7850.; r=.05; L=2.; P=1000.; Tq=500.
    p=chs_properties(r,E,nu,rho); nodes=np.array([[0,0,0],[L,0,0.]])
    K,M,_=assemble(nodes,[(0,1)],[p]); fixed=np.arange(6); F=np.zeros(12)
    F[7]=P; F[9]=Tq
    u,R,_=solve_static(K,F,fixed)
    exact_delta=P*L**3/(3*E*p['Iz']); exact_theta=P*L**2/(2*E*p['Iz']); exact_phi=Tq*L/(p['G']*p['J'])
    vals={"tip_v":(u[7],exact_delta),"tip_rz":(u[11],exact_theta),"tip_rx":(u[9],exact_phi)}
    return vals

def v1_second_plane():
    E=200e9; p=chs_properties(.05,E); L=2.; P=1000.; nodes=np.array([[0,0,0],[L,0,0.]])
    K,_,_=assemble(nodes,[(0,1)],[p]); F=np.zeros(12); F[8]=P
    u,_,_=solve_static(K,F,np.arange(6))
    return {"tip_w":(u[8],P*L**3/(3*E*p['Iy'])),"tip_ry":(u[10],-P*L**2/(2*E*p['Iy']))}

if __name__=='__main__':
    tests={**v1_cantilever(),**v1_second_plane()}; ok=True
    for name,(got,ex) in tests.items():
        e=rel(got,ex); print(f"{name:10s} got={got:.12e} exact={ex:.12e} relerr={e:.3e}"); ok &= e<1e-10
    raise SystemExit(0 if ok else 1)
