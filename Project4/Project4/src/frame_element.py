import numpy as np


def local_axes(x1, x2, reference=None):
    """Return 3x3 rotation whose rows are local unit axes in global coordinates."""
    x1 = np.asarray(x1, dtype=float); x2 = np.asarray(x2, dtype=float)
    dx = x2 - x1; L = np.linalg.norm(dx)
    if L <= 0:
        raise ValueError("zero-length frame element")
    ex = dx / L
    if reference is None:
        candidates = np.eye(3)
        ref = candidates[np.argmin(np.abs(candidates @ ex))]
    else:
        ref = np.asarray(reference, dtype=float)
        ref /= np.linalg.norm(ref)
        if abs(np.dot(ref, ex)) > 0.999999:
            raise ValueError("reference vector parallel to element")
    ez = np.cross(ex, ref); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    return np.vstack((ex, ey, ez)), L


def transformation(x1, x2, reference=None):
    R, L = local_axes(x1, x2, reference)
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = R
    return T, L


def local_stiffness(E, G, A, Iy, Iz, J, L):
    """Euler-Bernoulli 3D frame stiffness, local DOFs [u,v,w,rx,ry,rz] per node."""
    k = np.zeros((12,12))
    # axial and torsion
    ka = E*A/L; kt = G*J/L
    k[0,0]=k[6,6]=ka; k[0,6]=k[6,0]=-ka
    k[3,3]=k[9,9]=kt; k[3,9]=k[9,3]=-kt
    # bending v-rz, about local z (Iz)
    c1=12*E*Iz/L**3; c2=6*E*Iz/L**2; c3=4*E*Iz/L; c4=2*E*Iz/L
    ids=[1,5,7,11]
    kb=np.array([[c1,c2,-c1,c2],[c2,c3,-c2,c4],[-c1,-c2,c1,-c2],[c2,c4,-c2,c3]])
    k[np.ix_(ids,ids)] += kb
    # bending w-ry, about local y (Iy); signs follow right-handed rotations
    c1=12*E*Iy/L**3; c2=6*E*Iy/L**2; c3=4*E*Iy/L; c4=2*E*Iy/L
    ids=[2,4,8,10]
    kb=np.array([[c1,-c2,-c1,-c2],[-c2,c3,c2,c4],[-c1,c2,c1,c2],[-c2,c4,c2,c3]])
    k[np.ix_(ids,ids)] += kb
    return k


def local_consistent_mass(rho, A, Iy, Iz, J, L):
    """Consistent 3D Euler-Bernoulli frame mass incl. rotary torsional inertia."""
    M=np.zeros((12,12)); m=rho*A*L
    # axial
    M[np.ix_([0,6],[0,6])] = m/6*np.array([[2,1],[1,2]])
    # bending v-rz
    Mb=m/420*np.array([[156,22*L,54,-13*L],[22*L,4*L**2,13*L,-3*L**2],
                       [54,13*L,156,-22*L],[-13*L,-3*L**2,-22*L,4*L**2]])
    M[np.ix_([1,5,7,11],[1,5,7,11])] += Mb
    # bending w-ry (rotation sign convention differs)
    S=np.diag([1,-1,1,-1]); M[np.ix_([2,4,8,10],[2,4,8,10])] += S@Mb@S
    # torsional rotary inertia per length rho*J
    mt=rho*J*L
    M[np.ix_([3,9],[3,9])] += mt/6*np.array([[2,1],[1,2]])
    return M


def element_matrices(x1, x2, props, reference=None):
    T,L=transformation(x1,x2,reference)
    kl=local_stiffness(props['E'],props['G'],props['A'],props['Iy'],props['Iz'],props['J'],L)
    ml=local_consistent_mass(props['rho'],props['A'],props['Iy'],props['Iz'],props['J'],L)
    return T.T@kl@T, T.T@ml@T, T, L, kl

def local_geometric_stiffness_unit_compression(L):
    """Geometric stiffness per unit compressive axial force (P>0), Euler-Bernoulli."""
    kg=np.zeros((12,12))
    g=np.array([[36,3*L,-36,3*L],[3*L,4*L**2,-3*L,-L**2],
                [-36,-3*L,36,-3*L],[3*L,-L**2,-3*L,4*L**2]],float)/(30*L)
    kg[np.ix_([1,5,7,11],[1,5,7,11])] += g
    S=np.diag([1,-1,1,-1]); kg[np.ix_([2,4,8,10],[2,4,8,10])] += S@g@S
    return kg
