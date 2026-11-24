import sys,pathlib,numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from src.sections import chs_properties
from src.assembly import assemble,solve_static

def rel(a,b): return abs(a-b)/max(abs(b),1e-30)
def chain_nodes(L,n,R=None):
    x=np.linspace(0,L,n+1); pts=np.c_[x,np.zeros(n+1),np.zeros(n+1)]
    return pts if R is None else pts@R.T

def simply_supported_midspan(n=8,R=None):
    L=4.; w=1000.; p=chs_properties(.05); nodes=chain_nodes(L,n,R); elems=[(i,i+1) for i in range(n)]
    K,M,_=assemble(nodes,elems,[p]*n); F=np.zeros(6*(n+1)); dx=L/n
    # consistent uniform transverse nodal loads in global direction obtained by rotating local y if needed
    qdir=np.array([0.,1.,0.]) if R is None else R@np.array([0.,1.,0.])
    # exact consistent Euler-Bernoulli load for uniform local +y load
    fl=np.array([0,w*dx/2,0,0,0,w*dx**2/12, 0,w*dx/2,0,0,0,-w*dx**2/12.])
    for j in range(n):
        if R is None: fg=fl
        else:
            Te=np.zeros((12,12))
            for b in range(4): Te[3*b:3*b+3,3*b:3*b+3]=R.T
            fg=Te.T@fl
        F[6*j:6*j+12]+=fg
    # simple supports: translations at left all fixed; right local y,z fixed but axial free; rotations free
    # For rotated model impose constraints in rotated coordinates is awkward; transform full system back with block Q.
    if R is None:
        fixed=[0,1,2,3,6*n+1,6*n+2]
        u,_,_=solve_static(K,F,fixed); mid=n//2; disp=u[6*mid:6*mid+3]@qdir
    else:
        # block transform q_local = B q_global, B rows R.T? vector local=R.T global for R mapping base->rotated
        B=np.zeros_like(K.toarray())
        for i in range(2*(n+1)): B[3*i:3*i+3,3*i:3*i+3]=R.T
        Kl=B@K.toarray()@B.T; Fl=B@F
        fixed=[0,1,2,3,6*n+1,6*n+2]; free=np.setdiff1d(np.arange(len(F)),fixed); ul=np.zeros(len(F)); ul[free]=np.linalg.solve(Kl[np.ix_(free,free)],Fl[free]);
        mid=n//2; disp=ul[6*mid+1]
    exact=5*w*L**4/(384*p['E']*p['Iz']); return disp,exact

def frequency_test(n):
    L=4.; p=chs_properties(.05); nodes=chain_nodes(L,n); elems=[(i,i+1) for i in range(n)]; K,M,_=assemble(nodes,elems,[p]*n)
    # constrain axial/torsional and out-of-plane; simply-supported bending v with v ends zero, rz free
    fixed=[]
    for i in range(n+1): fixed += [6*i+0,6*i+2,6*i+3,6*i+4]
    fixed += [1,6*n+1]; free=np.setdiff1d(np.arange(K.shape[0]),np.unique(fixed))
    vals=eigh(K.toarray()[np.ix_(free,free)],M.toarray()[np.ix_(free,free)],subset_by_index=[0,0])[0]
    f=np.sqrt(vals[0])/(2*np.pi); mbar=p['rho']*p['A']; exact=np.pi/(2*L**2)*np.sqrt(p['E']*p['Iz']/mbar); return f,exact

def free_free_modes():
    p=chs_properties(.05); nodes=np.array([[0,0,0],[1,0,0],[2,.2,.1],[3,.4,-.1]],float); elems=[(0,1),(1,2),(2,3)]; K,M,_=assemble(nodes,elems,[p]*3)
    vals=eigh(K.toarray(),M.toarray(),eigvals_only=True); scale=max(vals[-1],1.); return np.sum(np.abs(vals)<1e-8*scale),vals[:8]

def balance_test():
    p=chs_properties(.05); nodes=chain_nodes(2.,2); K,_,_=assemble(nodes,[(0,1),(1,2)],[p,p]); F=np.zeros(18); F[13]=-1234.; fixed=list(range(6))+list(range(12,18)); u,R,_=solve_static(K,F,fixed); return R[fixed].sum()+F.sum()

if __name__=='__main__':
    d,e=simply_supported_midspan(); print('V2 midspan',d,e,rel(d,e))
    # Rodrigues rotation
    a=np.array([1.,2.,3.]); a/=np.linalg.norm(a); th=.73; A=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]); R=np.eye(3)*np.cos(th)+(1-np.cos(th))*np.outer(a,a)+np.sin(th)*A
    dr,er=simply_supported_midspan(R=R); print('V2 rotated',dr,er,rel(dr,er),'invariance',rel(dr,d))
    for n in [1,2,4,8,16]:
        f,fe=frequency_test(n); print('V3',n,f,fe,rel(f,fe))
    nz,vals=free_free_modes(); print('V5 zero modes',nz,'first eigs',vals)
    print('V5 balance residual',balance_test())
