import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from .frame_element import element_matrices

NDOF_NODE=6

def dofs_for_element(n1,n2):
    return np.r_[np.arange(6*n1,6*n1+6),np.arange(6*n2,6*n2+6)]

def assemble(nodes, elements, props_by_element):
    n=6*len(nodes); K=lil_matrix((n,n)); M=lil_matrix((n,n)); cache=[]
    for e,(n1,n2) in enumerate(elements):
        ke,me,T,L,kl=element_matrices(nodes[n1],nodes[n2],props_by_element[e])
        idx=dofs_for_element(n1,n2)
        for a,I in enumerate(idx):
            for b,J in enumerate(idx):
                K[I,J]+=ke[a,b]; M[I,J]+=me[a,b]
        cache.append((idx,T,L,kl))
    return csr_matrix(K),csr_matrix(M),cache

def solve_static(K,F,fixed):
    n=K.shape[0]; fixed=np.unique(np.asarray(fixed,dtype=int)); free=np.setdiff1d(np.arange(n),fixed)
    u=np.zeros(n); u[free]=spsolve(K[free][:,free],np.asarray(F)[free])
    reactions=K@u-np.asarray(F)
    return u,reactions,free
