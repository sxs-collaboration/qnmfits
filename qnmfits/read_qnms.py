import qnm
import numpy as np

_ksc = qnm.modes_cache

def qnm_from_tuple(tup, chi, M, s=-2):
    '''Get frequency and spherical_spheroidal mixing from qnm module
    
    Parameters
    ----------
    tup : tuple 
        Index (ell,m,n,sign) of QNM
    
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.
    
    M : float
        The mass of the black hole, M > 0.
    
    s : int, optional [Default: -2]
    
    Returns
    -------
    omega: complex
        Frequency of QNM. This frequency is the same units as arguments,
        as opposed to being in units of remnant mass.
    
    C : complex ndarray
        Spherical-spheroidal decomposition coefficient array
    
    ells : ndarray 
        List of ell values for the spherical-spheroidal mixing array
   
    '''
    ell, m, n, sign = tup
    if (sign == +1):
        mode_seq = _ksc(s, ell, m, n)
    elif (sign == -1):
        mode_seq = _ksc(s, ell, -m, n)
    else:
        raise ValueError("Last element of mode label must be "
                         "+1 or -1, instead got {}".format(sign))

    # The output from mode_seq is M*\omega
    try:
        Momega, _, C = mode_seq(chi, store=True)
    except:
        Momega, _, C = mode_seq(chi, interp_only=True)

    ells = qnm.angular.ells(s, m, mode_seq.l_max)

    if (sign == -1):
        Momega = -np.conj(Momega)
        C = (-1)**(ell + ells) * np.conj(C)

    # Convert from M*\omega to \omega
    omega = Momega/M
    return omega, C, ells
