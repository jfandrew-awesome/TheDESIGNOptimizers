import numpy as np


def chs_properties(r, E=200e9, nu=0.3, rho=7850.0):
    """Circular hollow section with wall thickness t=r/10. SI units."""
    r = float(r)
    if r <= 0:
        raise ValueError("outer radius must be positive")
    ri = 0.9 * r
    A = np.pi * (r**2 - ri**2)
    I = np.pi / 4.0 * (r**4 - ri**4)
    J = 2.0 * I
    G = E / (2.0 * (1.0 + nu))
    return {"A": A, "Iy": I, "Iz": I, "J": J, "E": E, "G": G, "rho": rho,
            "r": r, "ri": ri, "rg": np.sqrt(I/A)}
