
import numpy as np


def thomas_fermi(q, r, u_0=1, l=0):
    """
    Thomas-Fermi-like potential
    
    :param l: angular momentum
    :param q: potential range
    :param u_0:
    """
    
    tf0 = -1 * u_0 * np.exp(-1 * q * r) / r
    
    # to avoid numerical errors
    if l==0:
        return tf0
    else:
        return tf0 + l*(l+1) / (r**2 + 0.00001) 


def exponential_cosine(q, r, u_0=1):
    """ exponential cosine screened Coulomb interaction (EC) """
    return -1 * u_0 * np.exp(-1 * q * r) * np.cos(q * r)


def patterson_tf(q, r):
    """ Patterson function for Thomas-Fermi-like potential """
    return (2 * np.pi/q) * np.exp(-1 * q * r)


def square_well(a, r, V_0=1):
    """ Square wells potential """
    return 0 if r>a else -V_0
