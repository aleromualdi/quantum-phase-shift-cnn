
import numpy as np


def yukawa(q, r, V_0, l=0):
    """
    Yukawa potential.
    
    Parameters
    ----------
    `q`: float, potential range
    `r`: float, radius
    `V_0`: float, magnitude scaling constant
    `l`: int, angular momentum
    """
    
    tf0 = -1 * V_0 * np.exp(-1 * q * r) / r
    
    if l==0:
        return tf0
    else:
        return tf0 + l*(l+1) / (r**2)


def double_yukawa(q1, q2, r, V_01, V_02):
    """
    Double Yukawa potential.
    
    Parameters
    ----------
    `q1`: float, first compotent potential range
    `q2`: float, second compotent potential range
    `r`: float, radius
    `V_01`: float, first component magnitude scaling constant
    `V_02`: float, second component magnitude scaling constant
    """
    
    return -1 * V_01 * np.exp(-1* q1 * r) / r + V_02 * np.exp(-1* q2 * r) / r


def exponential_cosine(q, r, V_0):
    """
    Exponential cosine screened Coulomb interaction.

    Parameters
    ----------
    `q`: float, potential range
    `r`: float, radius
    `V_0`: float, magnitude scaling constant
    """

    return -1 * V_0 * np.exp(-1 * q * r) * np.cos(q * r) / r