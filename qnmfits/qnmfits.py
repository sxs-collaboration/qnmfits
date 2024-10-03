import numpy as np
import spherical_functions as sf

import scri
from scri import WaveformModes

from scipy.optimize import minimize
from scri.sample_waveforms import modes_constructor
from quaternion.calculus import indefinite_integral as integrate

from .read_qnms import qnm_from_tuple

def mismatch(h_A, h_B, t0, T, spherical_modes=None):
    """
    Returns the mismatch between two waveforms over the given spherical 
    harmonics, or over all modes.
    
    Parameters
    ----------
    h_A, h_B : WaveformModes
        The two waveforms to calculate the mismatch between.

    t0 : float
        The start time of the mismatch calculation.

    T : float
        The duration of the mismatch calculation, such that the end time of the
        mismatch integral is t0 + T. 

    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to compute mismatch over. If None, the 
        mismatch is calculated over all modes in the WaveformModes object.

    Returns
    -------
    mismatch : float
        The mismatch between the two waveforms.
    """

    if spherical_modes is None:

        numerator = np.real(h_A.inner_product(h_B, t1=t0, t2=t0+T))

        denominator = np.sqrt(np.real(
            h_A.inner_product(h_A, t1=t0, t2=T)*h_B.inner_product(h_B, t1=t0, t2=t0+T)
            ))
        
        return 1 - numerator/denominator

    else:

        h_A = h_A.copy()
        h_B = h_B.copy()

        for ell in range(h_A.ell_min, h_A.ell_max+1):
            for m in range(-ell, ell+1):
                if (ell, m) not in spherical_modes:
                    h_A.data[:, h_A.index(ell, m)] *= 0
                    h_B.data[:, h_B.index(ell, m)] *= 0

        numerator = np.real(h_A.inner_product(h_B, t1=t0, t2=t0+T))
        h_A_norm = np.real(h_A.inner_product(h_A, t1=t0, t2=t0+T))
        h_B_norm = np.real(h_B.inner_product(h_B, t1=t0, t2=t0+T))

        return 1 - numerator/np.sqrt(h_A_norm*h_B_norm)

        # h_A_h_B_inner_product = np.real(integrate(np.sum(h_A.data * np.conjugate(h_B.data), -1), h_A.t)[-1])
        # h_A_norm = integrate(np.sum(h_A.data * np.conjugate(h_A.data),-1), h_A.t)[-1].real
        # h_B_norm = integrate(np.sum(h_B.data * np.conjugate(h_B.data),-1), h_B.t)[-1].real
        # return 1 - h_A_h_B_inner_product / np.sqrt(h_A_norm * h_B_norm)

def qnm_modes(chi, M, mode_dict, dest=None, t_0=0., t_ref=0., **kwargs):
    """WaveformModes object with multiple qnms, 0 elsewhere

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.

    M : float
        The mass of the black hole, M > 0.

    mode_dict : dict
        Dict with keys in the format (ell_prime, m_prime, n) which is a QNM index,
        and values are a complex amplitude for that index.

    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the WaveformModes.data.
        Must be the correct shape.

    t_0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.

    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : WaveformModes
    """
    s = -2

    def data_functor(t, LM):

        d_shape = (t.shape[0], LM.shape[0])

        if (dest is None):
            # Need to allocate
            data = np.zeros(d_shape, dtype=complex)
        else:
            if ((dest.shape != d_shape)
                or (dest.dtype is not np.dtype(complex))):
                raise TypeError("dest has wrong dtype or shape")
            data = dest
            data.fill(0.)

        for (ell_prime, m_prime, n, sign), A in mode_dict.items():
            #print("Working on ({},{},{},{}) with A={}".format(ell_prime, m_prime, n, sign, A))
            omega, C, ells = qnm_from_tuple(
                (ell_prime, m_prime, n, sign), chi, M, s
                )
            
            expiwt = np.exp( complex(0., -1.) * omega * (t - t_ref))
            expiwt[ t < t_0 ] = 0.
            for _l, _m in LM:
                if (_m == m_prime):
                    c_l = C[ells == _l]
                    if (len(c_l) > 0):
                        c_l = c_l[0]
                        #print("Adding to ({},{}) with amplitude {}".format(_l, _m, c_l * A))
                        data[:, sf.LM_index(_l, _m, min(LM[:, 0]))] += c_l * A * expiwt

        return data
    
    constructor_statement = 'qnm_modes({0}, {1}, {2}, t_0={3}, t_ref={4}, **{5})'.format(
        chi, M, mode_dict, t_0, t_ref, kwargs
        )
    
    return modes_constructor(constructor_statement, data_functor, **kwargs)

def qnm_modes_as(chi, M, mode_dict, W_other, dest=None, t_0=0., t_ref=0., **kwargs):
    """
    WaveformModes object with multiple qnms, 0 elsewhere, with time
    and LM following W_other

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.

    M : float
        The mass of the black hole, M > 0.

    mode_dict : dict
        Dict with keys in the format (ell_prime, m_prime, n) which is a QNM index,
        and values are a complex amplitude for that index.

    W_other : WaveformModes object
        Get the time and LM from this WaveformModes object

    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the WaveformModes.data.
        Must be the correct shape.

    t_0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.

    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : WaveformModes
        A WaveformModes object filled with the model waveforms

    """

    t = W_other.t
    ell_min = W_other.ell_min
    ell_max = W_other.ell_max

    return qnm_modes(
        chi, 
        M, 
        mode_dict, 
        dest=dest, 
        t_0=t_0, 
        t_ref=t_ref,
        t=t, 
        ell_min=ell_min, 
        ell_max=ell_max,
        **kwargs
        )

def fit(W, chi, M,  mode_labels, spherical_modes=None, t_0=0., t_ref=0.):
    """
    Uses a modification of the mode limited eigenvalue method from 
    arXiv:2004.08347 to find best fit qnm amplitudes to a waveform.
    
    We modify the procedure by restricting the minimization of the mismatch
    only over modes that we care about. These modes are selected as follows:
    for every m, in mode_labels we find the maximum ell for which 
    some (ell, m, n , s) is in mode labels. This sets the ell_max for 
    that m. We don't consider modes with higher ell's in the mismatch
    to minimize. We are not trying to model these high-ell spherical modes, 
    so how well the low-ell qnm modes fit these high-ell modes through
    (spherical-spheroidal mixing) should not be part of the equation.
    Thus we project them out of the NR data and QNM modes.
    """
    
    res_mode_dict = {}
    for mode in mode_labels:
        res_mode_dict[mode] = None

    if spherical_modes is None:
        spherical_modes = [(l,m) for (l, m,_,_) in mode_labels]
    
    m_list = []
    [m_list.append(m) for (_, m,_,_) in mode_labels if m not in m_list]
    # Break problem into one m at a time. The m's are decoupled, and the truncation
    # in ell for each m is different.
    for m in m_list:
        mode_labels_m = [label for label in mode_labels if label[1]==m]
        ell_list = [l for l,em in spherical_modes if em==m]
        ell_max_m = max(ell_list)  # Truncate all modes above ell_max_m for this m
        data_index_m = [sf.LM_index(l, m, W.ell_min) for l in ell_list]
        
        A = np.zeros((len(W.t),len(spherical_modes)), dtype=complex) # Data overlap with qnm modes
        B = np.zeros((len(W.t),len(spherical_modes), len(mode_labels_m)), dtype=complex)
        
        W_trunc = W[:,:ell_max_m+1]
        A = W_trunc.data[:, data_index_m]
        for mode_index, label in enumerate(mode_labels_m):
            tmp_mode_dict = {label: 1.}
            Q = qnm_modes_as(chi, M, tmp_mode_dict, W_trunc, 
                             t_0=t_0, t_ref=t_ref)
            B[:,:,mode_index] = Q.data[:,data_index_m]

        A = np.reshape(A, len(W.t)*len(data_index_m))
        B = np.reshape(B, (len(W.t)*len(data_index_m), len(mode_labels_m)))
        C = np.linalg.lstsq(B, A, rcond=None)
        for i, label in enumerate(mode_labels_m):
            res_mode_dict[label] = C[0][i]
        
    return res_mode_dict

def fit_chi_M_and_modes(W, mode_labels, spherical_modes=None, t_0=0., t_ref=0., 
                        maxiter=1000, xtol=1e-8, ftol=1e-8):
    
    """Use scipy.optimize.minimize to find best fit spin, mass, and qnm amplitudes to a waveform
    Parameters
    ----------
    W : WaveformModes
    
    mode_labels : list of tuples
    
    spherical_modes : list, optional [Default: None]
        List of tuples (l,m) over which to minimize mismatch over. If 'None',
        then the (l,m) modes of the mode_labels will be used.
    
    t_0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.
    
    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.
    
    Returns
    -------
    chi : double
    
    M : double
    
    res_mode_dict : dict
    
    res : scipy.optimize.OptimizeResult
    """

    dest_Q = np.zeros(W.data.shape, dtype=complex)
    dest_diff = np.zeros(W.data.shape, dtype=complex)

    W_fitted_modes = W.copy()
    W_fitted_modes.data *= 0.
    for LM_mode in list(set([(mode[0], mode[1]) for mode in mode_labels])):
        W_fitted_modes.data[:,sf.LM_index(LM_mode[0], LM_mode[1], W_fitted_modes.ell_min)] = W.data[:,sf.LM_index(LM_mode[0], LM_mode[1], W.ell_min)]
    
    def goodness(chi_M, *args):
        chi = chi_M[0]
        M = chi_M[1]
        if chi < 0.0 or chi > 0.99 or M < 0.0 or M > 1.0:
            return 1e6
        mode_dict = fit(W_fitted_modes, chi, M, mode_labels,
                spherical_modes=spherical_modes, t_0=t_0, t_ref=t_ref)
        Q = qnm_modes_as(chi, M, mode_dict, W_fitted_modes)
        Q_fitted_modes = Q.copy()
        Q_fitted_modes.data *= 0.
        for LM_mode in list(set([(mode[0], mode[1]) for mode in mode_labels])):
            Q_fitted_modes.data[:,sf.LM_index(LM_mode[0], LM_mode[1], Q_fitted_modes.ell_min)] = Q.data[:,sf.LM_index(LM_mode[0], LM_mode[1], Q.ell_min)]
        diff = W_fitted_modes.copy()
        diff.data -= Q_fitted_modes.data
        return integrate(diff.norm(), diff.t)[-1]

    x0 = [.7, .95]

    bnds = [(0., .999), (.7, 1.)]
    bnds = tuple(bnds)

    res = minimize(goodness, x0,  method='Nelder-Mead', bounds=bnds, options={'maxiter': maxiter, 'maxfev': maxiter, 'xatol': xtol, 'fatol': ftol})
    if (res.success):
        chi = res.x[0]
        M = res.x[1]
        res_mode_dict = fit(W, chi, M, mode_labels,
                            spherical_modes=spherical_modes, t_0=t_0, t_ref=t_ref)
        return chi, M, res_mode_dict, res
    else:
        return None, None, None, res


# Greedy-fit functions
# --------------------
    
def mode_power_order(W, topN=10, t_0=-np.Inf):
    """Return a list of topN of W's mode indices, sorted by power per mode

    Parameters
    ----------
    W : WaveformModes

    topN : int, optional [Default: 10]

    t_0 : float, optional [Default: -np.Inf]
        Only compute power after t_0

    Returns
    -------
    mode_list : ndarray
      List of [l, m] indices, where the 0th row is the mode with the most power
    """
    sliced = W.data[W.t > t_0, :]
    return W.LM[np.argsort(np.sum(np.square(np.abs(sliced)),axis=0))][-1:-1-topN:-1]

def add_modes(modes_so_far_dict, loudest_lms, n_max=7, retrograde=False):
    """
    Add one or two modes to the list of modes_so_far, based on the
    loudest_lm spherical harmonic mode.  The newly added mode will
    have the lowest overtone number which is not yet present in
    modes_so_far as long as n<n_max.  If retrograde is False, only
    prograde modes are added except when m is 0, where both 
    both prograde and retrograde modes are added. If retrograde is 
    True, both prograde and retrograde modes are always added

    Parameters
    ----------
    modes_so_far_dict: dictionary
         Dictionary keys are tuples (l,m,n,sign) and values are complex amplitudes.
         
    loudest_lms:  list of tuples (l,m)
      *Spherical* harmonic indices in order of loudest first
    
    n_max: int, optional [Default: 7]
        Maximum overtone number to include in fits (includes n_max).
    
    retrograde: boolean, optional [Default: False]
        All retrograde QNMs included if True, else only prograde mode is added
        except for m=0, where both are always added.
         
    Returns
    -------
    Dictionary
    """

    new_modes = modes_so_far_dict
    #Loop through loudest_lms till we find a mode we can add
    for loudest_l, loudest_m in loudest_lms:
        ns_so_far = list({n for (l, m, n, _) in modes_so_far_dict.keys()
                          if ((l==loudest_l) and (m==loudest_m))})

        max_n_so_far = np.max(ns_so_far) if len(ns_so_far)>0 else -1
        new_n = max_n_so_far + 1
        
        if new_n <= n_max:
            if (loudest_m == 0) or retrograde:
                new_modes[(loudest_l, loudest_m, new_n, +1)] = None
                new_modes[(loudest_l, loudest_m, new_n, -1)] = None
            else:
                new_modes[(loudest_l, loudest_m, new_n,
                           int(np.sign(loudest_m)))] = None
            break
        elif new_n > n_max and (loudest_l, loudest_m) == (loudest_lms[-1][0], loudest_lms[-1][1]):
            print("Cannot find a valid mode to add.")
    return new_modes

def pick_nmodes_greedy(W, chi, M, target_frac, num_modes_max, 
                       nmodes_to_report=None, initial_modes_dict={}, t_0=0.,
                       t_ref=0., n_max=7, interpolate=True, use_news_power=True,
                       retrograde=False):
    """Calculates the fraction of unmodeled power and mismatch for each number
    of modes in nmodes_to_report. By default, the power is calculated using the News
    function. 

    Parameters
    ----------
    W: WaveformModes

    chi: float

    M: float

    target_frac: float

    num_modes_max: int

    nmodes_to_report: list, optional [Default: None]
        List of modes to append on the way to num_modes_max.

    initial_modes_dict: dictionary, optional [Default: {}]
        Dictionary keys are tuples (l,m,n,sign) and values are complex amplitudes.

    t_0: float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.

    t_ref: float, optional [Default: 0.]
        Time at which amplitudes are specified.

    n_max: int, optional [Default: 7]
        Maximum overtone number to allow for a given mode (includes n_max).

    retrograde: boolean, optional [Default: False]
        All retrograde QNMs included if True, else only prograde mode is added
        except for m=0, where both are always added.

    Returns
    -------
    mode_dict: dict from fit_W_modes

    Q: WaveformModes
      QNM model waveform corresponding to res_mode_dict

    diff: WaveformModes
      Difference waveform W-Q

    frac_unmodeled_powers: list
      Fraction of power still unmodeled for each nmode_to_report

    wf_mismatches: list
      Mismatches between W and Q for each nmode_to_report
    """

    # Make sure that target_frac is between 0 and 1,
    # Make sure that num_modes_max is a non-negative integer
    if not ((0. <= target_frac) or (target_frac <= 1.)):
        raise ValueError("target_frace={} should be "
                         "between 0 and 1.".format(target_frac))

    if not (num_modes_max > 0):
        raise ValueError("num_modes_max={} must be "
                         "greater than 0.".format(num_modes_max))

    if None in initial_modes_dict.values():
        raise ValueError("All values in initial_modes_dict should be complex "
                         "numbers.")

    if nmodes_to_report is None:
        nmodes_to_report = []

    # Make sure that the waveform starts at t_0, otherwise
    # inner_product returns faulty values due to interpolation issues
    if interpolate:
        W = W[np.argmin(abs(W.t - t_0)):]
    
    if use_news_power:
        if W.dataType == scri.h:
            W_power_waveform = W.copy()
            W_power_waveform.data = W.data_dot
            W_power_waveform.dataType = scri.hdot
        elif W.dataType == scri.hdot:
            W_power_waveform = W.copy()
        else:
            raise ValueError("W is not of type scri.h or scri.hdot")
    else:
        W_power_waveform = W.copy()
    W_power = np.real(W_power_waveform.inner_product(W_power_waveform, t1=t_0))
        
    # Initially, the difference between the waveform and model is just
    # the waveform itself.
    diff = W_power_waveform.copy()

    mode_dict = initial_modes_dict
    frac_unmodeled_powers = []
    wf_mismatches = []
    mode_dicts = []
    for i_mode in np.arange(0, num_modes_max):

        # Add one or two modes
        num_modes = len(W.LM)
        loudest_lms = mode_power_order(diff, topN=num_modes, t_0=t_0)
            
        mode_dict = add_modes(mode_dict, loudest_lms, n_max, retrograde)

        # Build a ringdown model
        mode_labels = list(mode_dict.keys())
        mode_dict = fit_W_modes(W, chi, M, mode_labels, spherical_modes=None, t_0=t_0, t_ref=t_ref)
        
        Q = qnm_modes_as(chi, M, mode_dict, W, t_0=t_0, t_ref=t_ref)

        # How much power is unmodeled?
        if use_news_power:
            if W.dataType == scri.h:
                np.subtract(W.data, Q.data, out=diff.data)
                diff.data = diff.data_dot
            elif W.dataType == scri.hdot:
                np.subtract(W.data, Q.data, out=diff.data)
        else:
            np.subtract(W.data, Q.data, out=diff.data)
        diff_power = np.real(diff.inner_product(diff, t1=t_0))
        frac_unmodeled_power = diff_power / W_power

        if i_mode+1 in nmodes_to_report:
            frac_unmodeled_powers.append(frac_unmodeled_power)
            wf_mismatches.append(waveform_mismatch(W, Q, t_0=t_0))
            mode_dicts.append(mode_dict)

        if (frac_unmodeled_power < target_frac):
            break # Don't need to add any more modes

    frac_unmodeled_powers.append(frac_unmodeled_power)
    wf_mismatches.append(waveform_mismatch(W, Q, t_0=t_0))
    mode_dicts.append(mode_dict)
    # We've hit the max number of modes
    return mode_dicts, Q, diff, frac_unmodeled_powers, wf_mismatches

def pick_modes_greedy(W, chi, M, target_frac, num_modes_max, 
                      initial_modes_dict={}, t_0=0., t_ref=0., n_max=7, 
                      interpolate=True, use_news_power=True, retrograde=False):
    """Calculates the fraction of unmodeled power and mismatch for the
    num_modes_max number of modes.
    """
    mode_dict, Q, diff, power, mismatch = pick_nmodes_greedy(
        W, 
        chi, 
        M, 
        target_frac,
        num_modes_max, 
        nmodes_to_report=None,
        initial_modes_dict=initial_modes_dict,
        t_0=t_0, 
        t_ref=t_ref,
        n_max=n_max,
        interpolate=interpolate,
        use_news_power=use_news_power,
        retrograde=retrograde)
    
    return mode_dict[0], Q, diff, power[0], mismatch[0]
