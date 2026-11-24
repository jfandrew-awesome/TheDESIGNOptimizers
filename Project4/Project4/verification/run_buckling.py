import sys,pathlib,numpy as np
from scipy.linalg import eigvalsh
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from src.sections import chs_properties
from src.frame_element import element_matrices,local_geometric_stiffness_unit_compression
from src.assembly import dofs_for_element

def test(n):
    L=3.; p=chs_properties(.05); nn=n+1; K=np.zeros((6*nn,6*nn)); G=np.zeros_like(K); x=np.linspace(0,L,nn)
    for e in range(n):
        ke,_,T,Le,_=element_matrices([x[e],0,0],[x[e+1],0,0],p); ge=T.T@local_geometric_stiffness_unit_compression(Le)@T; ids=dofs_for_element(e,e+1)
        K[np.ix_(ids,ids)]+=ke; G[np.ix_(ids,ids)]+=ge
    # pinned-pinned in one bending plane: retain v,rz; v=0 at ends
    active=[]
    for i in range(nn): active += [6*i+1,6*i+5]
    fixed=[1,6*n+1]; free=np.setdiff1d(active,fixed)
    vals=eigvalsh(K[np.ix_(free,free)],G[np.ix_(free,free)]); P=vals[vals>1e-5][0]; ex=np.pi**2*p['E']*p['Iz']/L**2
    return P,ex,abs(P-ex)/ex
if __name__=='__main__':
    for n in [1,2,4,8]: print('V4',n,*test(n))
