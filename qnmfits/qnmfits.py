import numpy as np
import spherical_functions as sf

import scri
import bisect

from scipy.optimize import minimize
from scri.sample_waveforms import modes_constructor
from quaternion.calculus import indefinite_integral as integrate

from .read_qnms import qnm_from_tuple


def mismatch(h_A, h_B, t0, T, spherical_modes=None):
    """
    Returns the mismatch between two waveforms over the given spherical
    harmonics or over all modes.

    Parameters
    ----------
    h_A, h_B : WaveformModes
        The two waveforms to calculate the mismatch between.

    t0 : float
        The start time of the mismatch calculation.

    T : float
        The duration of the mismatch calculation, such that the end time of the
        mismatch integral is t0 + T.

    spherical_modes : list of tuples, optional [Default: None]
        A sequence of (ell, m) modes to compute the mismatch over. If None, the
        mismatch is calculated over all modes in the WaveformModes object.

    Returns
    -------
    mismatch : float
        The mismatch between the two waveforms.
    """
    if spherical_modes is None:

        numerator = np.real(h_A.inner_product(h_B, t1=t0, t2=t0+T))

        denominator = np.sqrt(np.real(
            h_A.inner_product(h_A, t1=t0, t2=T)
            * h_B.inner_product(h_B, t1=t0, t2=t0+T)
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


def qnm_WaveformModes(times, chif, Mf, qnm_amps, t0=0, t_ref=None, ell_min=2,
                      ell_max=8, t0_method='geq'):
    """
    Construct a WaveformModes object from a set of QNM amplitudes.

    Parameters
    ----------
    times : ndarray
        The times at which the waveform is defined.

    chif : float
        The magnitude of the dimensionless spin of the black hole, which along
        with Mf determines the QNM frequencies.

    Mf : float
        The mass of the remnant black hole.

    qnm_amps : dict
        Dictionary of QNM amplitudes. The keys are tuples (ell, m, n, sign)
        and the values are complex amplitudes.

    t0 : float, optional [Default: 0]
        The ringdown model start time.

    t_ref : float, optional [Default: None]
        The time at which the QNM amplitudes are defined. If None, t_ref = t0.

    ell_min : int, optional [Default: 2]
        The minimum ell value to include in the WaveformModes object.

    ell_max : int, optional [Default: 8]
        The maximum ell value to include in the WaveformModes object.

    t0_method : str, optional [Default: geq]
        A requested ringdown start time will in general lie between times on
        the default time array. There are different approaches to deal with
        this, which can be specified here.

        Options are:

            - 'geq'
                The waveform will be non-zero for times greater than or equal
                to t0.

    """

    # If a reference time for the QNM amplitudes isn't given, use t0
    if t_ref is None:
        t_ref = t0

    # The spherical-harmonic mode (ell, m) indices for the requested ell_min
    # and ell_max
    ell_m_list = sf.LM_range(ell_min, ell_max)

    # Initialize the data array
    data = np.zeros((len(times), len(ell_m_list)), dtype=complex)

    # Loop through each provided QNM amplitude, and populate the data array
    # with the sum of (appropriately weighted) QNMs
    for (ell, m, n, sign), A in qnm_amps.items():

        # Get the complex QNM frequencies omega and the spherical-spheroidal
        # mixing coefficients C. C is a list of mixing coefficients,
        # corresponding to different ell values (given by C_ells).
        omega, C, C_ells = qnm_from_tuple((ell, m, n, sign), chif, Mf)

        # Construct the pure QNM damped sinusoid. This has not yet been
        # weighted by the spherical-spheroidal mixing coefficients.
        h_qnm = A*np.exp(-1j*omega*(times - t_ref))

        if t0_method == 'geq':
            h_qnm[times < t0] = 0

        # Loop through each spherical-harmonic mode in the final data array
        for ell_prime, m_prime in ell_m_list:

            # h_qnm will only mix into spherical-harmonic modes with the same m
            if (m_prime == m):

                # Get the mixing coefficient between the (ell, m, n) QNM and
                # the (ell', m') sppherical-harmonic mode
                C_ell = C[C_ells == ell_prime][0]

                # Add the weighted QNM to the appropriate spherical mode
                data[:, sf.LM_index(ell_prime, m_prime, ell_min)] += \
                    C_ell*h_qnm

    # Construct the WaveformModes object
    wm = scri.WaveformModes(
        dataType=scri.h,
        t=times,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True
    )

    return wm


def fit(data, chif, Mf, qnms, spherical_modes=None, t0=0, T=100, t_ref=None,
        t0_method='geq'):
    """
    Find the best-fit (as determined by a least-squares fit) complex QNM
    amplitudes of a ringdown model, fitted to some data. The data is a
    WaveformModes object, decomposed into spin-weighted spherical harmonics.
    The fit can be performed to any subset of the spherical modes, and the
    spherical-spheroidal mixing coefficients are used to correctly weight the
    QNM amplitudes for each spherical mode.

    The ringdown spectrum is fixed accoring to the provided chif and Mf values.

    Parameters
    ----------
    data : WaveformModes
        Waveform to use for fitting QNM amplitudes.

    chif : float
        The magnitude of the dimensionless spin of the remnant black hole.
        Along with M, this determines the QNM frequencies.

    Mf : float
        The mass of the remnant black hole. This is the factor which the QNM
        frequencies are divided through by, and so determines the units of
        the returned quantity.

        When working with numerical-relativity simulations, we work in units
        scaled by the total mass of the binary system, M. In this case,
        providing the dimensionless Mf value (the final mass scaled by the
        total binary mass) will ensure the QNM frequencies are in the correct
        units (that is, scaled by the total binary mass). This is because the
        frequencies loaded from file are scaled by the remnant black hole mass
        (Mf*omega). So, by dividing by the remnant black hole mass scaled by
        the total binary mass (Mf/M), we are left with
        Mf*omega/(Mf/M) = M*omega.

    qnms : list of (ell, m, n, sign) tuples
        List of quasinormal modes to include in the ringdown model. For regular
        (positive real part) modes use sign=+1. For mirror (negative real part)
        modes use sign=-1.

    spherical_modes : list of (ell, m) tuples, optional [Default: None]
        List of spherical-harmonic modes to fit the ringdown model to. If None,
        the fit is performed over all available spherical modes in the data.

    t0 : float, optional [Default: 0]
        The ringdown model start time.

    T : float, optional [Default: 100]

    t_ref : float, optional [Default: None]
        The time at which the QNM amplitudes are defined. If None, t_ref = t0.

    t0_method : str, optional [Default: geq]
        A requested ringdown start time will in general lie between times on
        the default time array (the same is true for the end time of the
        analysis). There are different approaches to deal with this, which can
        be specified here.

        Options are:

            - 'geq'
                Take data at times greater than or equal to t0. Note that
                we still treat the ringdown start time as occuring at t0,
                so the best fit coefficients are defined with respect to
                t0.

            - 'closest'
                Identify the data point occuring at a time closest to t0,
                and take times from there.

    Returns
    -------
    result : dict
        A dictionary of useful information related to the fit. Keys include:

            - 'mismatch' : float
                The mismatch between the best-fit ringdown waveform and data,
                computed over the requested spherical_modes.
            - 'amplitudes' : dict
                The best-fit complex amplitudes. There is a complex amplitude
                for each ringdown mode.
            - 'frequencies' : dict
                The complex QNM frequencies used in the model. There is a
                complex frequency for each ringdown mode.
            - 'data' : WaveformModes
                The (masked) data used in the fit.
            - 'model': WaveformModes
                The best-fit model waveform.
    """
    if t_ref is None:
        t_ref = t0

    result = {}

    # Dictionary to store the best-fit QNM amplitudes
    qnm_amps = {}

    # Dictionary to store the complex QNM frequencies
    qnm_freqs = {}

    # Default to all available spherical modes if None
    if spherical_modes is None:
        spherical_modes = sf.LM_range(data.ell_min, data.ell_max)

    # List of all the m indices we're considering in this fit
    m_list = []
    for (_, m, _, _) in qnms:
        if m not in m_list:
            m_list.append(m)

    # Window the data
    if t0_method == 'geq':
        start_index = bisect.bisect_left(data.t, t0)
        end_index = bisect.bisect_left(data.t, t0+T) - 1
        h_cut = data.copy()[start_index:end_index, :]

    elif t0_method == 'closest':
        h_cut = data.copy()[
            np.argmin(abs(data.t - t0)):np.argmin(abs(data.t - t0 - T))+1, :
        ]

    else:
        print("""Requested t0_method is not valid. Please choose between 'geq'
              and 'closest'""")

    # Break problem into one m at a time. The m's are decoupled, and the
    # truncation in ell for each m is different.
    for m in m_list:

        # The QNMs with the current value of m
        qnms_m = [label for label in qnms if label[1] == m]

        # The spherical-harmonic modes with the current value of m
        spherical_modes_m = [
            (ell, mp) for ell, mp in spherical_modes if mp == m
        ]

        # The ell indices in the spherical-harmonic modes with the current
        # value of m
        spherical_ell_list = [ell for ell, _ in spherical_modes_m]

        # Truncate all modes above the maximum ell for this m (when we index
        # without .data, we index the ells directly)
        h_trunc = h_cut[:, :max(spherical_ell_list)+1]

        # Index the requested spherical-harmonic modes (for this m) from the
        # data. b has shape (len(h_cut.t), len(spherical_modes_m)).
        data_index_m = [h_trunc.index(ell, m) for ell in spherical_ell_list]
        b = h_trunc.data[:, data_index_m]

        # Initialize the design (or coefficient) matrix for the least-squares
        # fit
        a = np.zeros(
            (len(h_cut.t), len(spherical_modes_m), len(qnms_m)),
            dtype=complex
        )

        # Loop over each QNM in our model with the current value of m. For
        # efficiency, we compute the QNM data once and then weight it by the
        # spherical-spheroidal mixing coefficients.
        for qnm_index, label in enumerate(qnms_m):

            # Initialize the data array for the current QNM. We will store
            # weighted copies of the QNM in this array, one for each
            # spherical-harmonic mode. This will then be used to populate the
            # design matrix.
            qnm_data = np.zeros(
                (len(h_cut.t), len(spherical_modes_m)),
                dtype=complex
            )

            # Get the complex QNM frequencies omega and the spherical-
            # spheroidal mixing coefficients C. C is a list of mixing
            # coefficients, corresponding to different ell values (given by
            # C_ells).
            omega, C, C_ells = qnm_from_tuple(label, chif, Mf)

            qnm_freqs[label] = omega

            # Construct the pure QNM damped sinusoid. This has not yet been
            # weighted by the spherical-spheroidal mixing coefficients.
            h_qnm = np.exp(-1j*omega*(h_cut.t - t0))

            # Loop through each spherical-harmonic mode in the final data array
            for i, ell_prime in enumerate(spherical_ell_list):

                # Get the mixing coefficient between the (ell, m, n) QNM and
                # the (ell', m') sppherical-harmonic mode
                C_ell = C[C_ells == ell_prime][0]

                # Store the weighted QNM
                qnm_data[:, i] += C_ell*h_qnm

            # Store the weighted QNMs in the design matrix
            a[:, :, qnm_index] = qnm_data

        # Reshape the arrays for np.lstsq

        # The data array needs to be one-dimensional. You can view this as
        # joining each hlm array to make a single long timeseries.
        b = np.reshape(b, len(h_cut.t)*len(data_index_m))

        # We similarly collapse the spherical-harmonic modes axis in the
        # design matrix
        a = np.reshape(a, (len(h_cut.t)*len(data_index_m), len(qnms_m)))

        # Perform the fit
        amplitudes, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        # Store the complex QNM amplitudes. Note that we re-scale them to t_ref
        # (the time at which the amplitudes are defined).
        for i, label in enumerate(qnms_m):
            qnm_amps[label] = \
                amplitudes[i]*np.exp(-1j*qnm_freqs[label]*(t_ref-t0))

    result['amplitudes'] = qnm_amps

    # Compute the best-fit waveform
    qnm_wm = qnm_WaveformModes(
        h_cut.t,
        chif,
        Mf,
        qnm_amps,
        t0=t0,
        t_ref=t_ref,
        ell_min=h_cut.ell_min,
        ell_max=h_cut.ell_max
    )
    result['model'] = qnm_wm

    # Also store the mismatch between the best-fit waveform and the data
    result['mismatch'] = mismatch(
        h_cut,
        qnm_wm,
        t0,
        T,
        spherical_modes
    )

    # Store the data used in the fit
    result['data'] = h_cut

    # Store the complex QNM frequencies
    result['frequencies'] = qnm_freqs

    return result


def qnm_modes(chif, M, mode_dict, dest=None, t0=0., t_ref=0., **kwargs):
    """
    WaveformModes object with multiple qnms, 0 elsewhere.

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chif : float
        The dimensionless spin of the black hole, 0. <= chif < 1.

    M : float
        The mass of the black hole, M > 0.

    mode_dict : dict
        Dict with keys in the format (l, m, n, sign) which is a QNM index,
        and values are a complex amplitude for that index.

    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the WaveformModes.data.
        Must be the correct shape.

    t0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t0.

    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : WaveformModes object
    """
    s = -2

    def data_functor(t, LM):

        d_shape = (t.shape[0], LM.shape[0])

        if (dest is None):
            # Need to allocate
            data = np.zeros(d_shape, dtype=complex)
        else:
            if (
                (dest.shape != d_shape) or
                (dest.dtype is not np.dtype(complex))
            ):
                raise TypeError("dest has wrong dtype or shape")
            data = dest
            data.fill(0.)

        for (ell_prime, m_prime, n, sign), A in mode_dict.items():
            omega, C, ells = qnm_from_tuple(
                (ell_prime, m_prime, n, sign), chif, M, s
            )

            expiwt = np.exp(complex(0., -1.) * omega * (t - t_ref))
            expiwt[t < t0] = 0.
            for _l, _m in LM:
                if (_m == m_prime):
                    c_l = C[ells == _l]
                    if (len(c_l) > 0):
                        c_l = c_l[0]
                        data[:, sf.LM_index(_l, _m, min(LM[:, 0]))] += \
                            c_l * A * expiwt

        return data

    constructor_statement = \
        'qnm_modes({0}, {1}, {2}, t0={3}, t_ref={4}, **{5})'.format(
            chif, M, mode_dict, t0, t_ref, kwargs
        )

    return modes_constructor(constructor_statement, data_functor, **kwargs)


def qnm_modes_as(chif, M, mode_dict, W_other, dest=None, t0=0., t_ref=0.,
                 **kwargs):
    """
    WaveformModes object with multiple qnms, 0 elsewhere, with time
    and LM following W_other.

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chif : float
        The dimensionless spin of the black hole, 0. <= chif < 1.

    M : float
        The mass of the black hole, M > 0.

    mode_dict : dict
        Dict with keys in the format (l, m, n, sign) which is a QNM index,
        and values are a complex amplitude for that index.

    W_other : WaveformModes object
        Get the time and LM from this WaveformModes object

    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the WaveformModes.data.
        Must be the correct shape.

    t0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t0.

    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : WaveformModes object
        A WaveformModes object filled with the model waveform
    """
    t = W_other.t
    ell_min = W_other.ell_min
    ell_max = W_other.ell_max

    return qnm_modes(
        chif,
        M,
        mode_dict,
        dest=dest,
        t0=t0,
        t_ref=t_ref,
        t=t,
        ell_min=ell_min,
        ell_max=ell_max,
        **kwargs
    )


def fit_chif_M_and_modes(W, qnms, spherical_modes=None, t0=0., t_ref=0., 
                        maxiter=1000, xtol=1e-8, ftol=1e-8):    
    """
    Use scipy.optimize.minimize to find best fit spin, mass, and QNM amplitudes
    of a waveform.
    
    Parameters
    ----------
    W : WaveformModes object
        Waveform to use for fitting spin, mass, and QNM amplitudes
    
    qnms : list of tuples (l, m, n, sign)
        List of modes to fit over. 

    spherical_modes : list of tuples (l,m), optional [Default: None]
        A sequence of (l,m) modes to fit over. If None, all (l,m) modes in
        model_labels are used.
    
    t0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t0.
    
    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    maxiter : int, optional [Default: 1000]

    xtol : float, optional [Default: 1e-8]

    ftol : float, optional [Default: 1e-8]
    
    Returns
    -------
    chif : double
        Best-fit spin.
    
    M : double
        Best-fit mass.
    
    res_mode_dict : dict
        Dictionary of QNM modes with complex amplitudes.
    
    res : scipy.optimize.OptimizeResult
    """

    dest_Q = np.zeros(W.data.shape, dtype=complex)
    dest_diff = np.zeros(W.data.shape, dtype=complex)

    W_fitted_modes = W.copy()
    W_fitted_modes.data *= 0.
    for LM_mode in list(set([(mode[0], mode[1]) for mode in qnms])):
        W_fitted_modes.data[:,sf.LM_index(LM_mode[0], LM_mode[1], W_fitted_modes.ell_min)] = W.data[:,sf.LM_index(LM_mode[0], LM_mode[1], W.ell_min)]
    
    def goodness(chif_M, *args):
        chif = chif_M[0]
        M = chif_M[1]
        if chif < 0.0 or chif > 0.99 or M < 0.0 or M > 1.0:
            return 1e6
        mode_dict = fit(W_fitted_modes, chif, M, qnms,
                spherical_modes=spherical_modes, t0=t0, t_ref=t_ref)
        Q = qnm_modes_as(chif, M, mode_dict, W_fitted_modes)
        Q_fitted_modes = Q.copy()
        Q_fitted_modes.data *= 0.
        for LM_mode in list(set([(mode[0], mode[1]) for mode in qnms])):
            Q_fitted_modes.data[:,sf.LM_index(LM_mode[0], LM_mode[1], Q_fitted_modes.ell_min)] = Q.data[:,sf.LM_index(LM_mode[0], LM_mode[1], Q.ell_min)]
        diff = W_fitted_modes.copy()
        diff.data -= Q_fitted_modes.data
        return integrate(diff.norm(), diff.t)[-1]

    x0 = [.7, .95]

    bnds = [(0., .999), (.7, 1.)]
    bnds = tuple(bnds)

    res = minimize(goodness, x0,  method='Nelder-Mead', bounds=bnds, options={'maxiter': maxiter, 'maxfev': maxiter, 'xatol': xtol, 'fatol': ftol})
    if (res.success):
        chif = res.x[0]
        M = res.x[1]
        res_mode_dict = fit(W, chif, M, qnms,
                            spherical_modes=spherical_modes, t0=t0, t_ref=t_ref)
        return chif, M, res_mode_dict, res
    else:
        return None, None, None, res


# Greedy-fit functions
# --------------------
    
def mode_power_order(W, topN=10, t0=-np.Inf):
    """Returns a list of topN indices sorted by power per mode for a waveform.

    Parameters
    ----------
    W : WaveformModes object

    topN : int, optional [Default: 10]

    t0 : float, optional [Default: -np.Inf]
        Only compute power after t0.

    Returns
    -------
    mode_list : ndarray
      List of (l,m) indices, where the 0th row is the mode with the most
      power.
    """
    sliced = W.data[W.t > t0, :]
    return W.LM[np.argsort(np.sum(np.square(np.abs(sliced)),axis=0))][-1:-1-topN:-1]

def add_modes(modes_so_far_dict, loudest_lms, n_max=7, retrograde=False):
    """
    Add one or two modes to the list of modes_so_far, based on the
    loudest_lm spherical harmonic mode. The newly added mode will
    have the lowest overtone number which is not yet present in
    modes_so_far as long as n<n_max. If retrograde is False, only
    prograde modes are added except when m is 0, where both 
    both prograde and retrograde modes are added. If retrograde is 
    True, both prograde and retrograde modes are always added.

    Parameters
    ----------
    modes_so_far_dict : dict
         Dictionary keys are tuples (l,m,n,sign) and values are complex amplitudes.
         
    loudest_lms : list of tuples (l,m)
      *Spherical* harmonic indices in order of loudest first
    
    n_max : int, optional [Default: 7]
        Maximum overtone number to include in fits (includes n_max).
    
    retrograde : boolean, optional [Default: False]
        All retrograde QNMs included if True, else only prograde mode is added
        except for m=0, where both are always added.
         
    Returns
    -------
    new_modes : dict
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

def pick_nmodes_greedy(W, chif, M, target_frac, num_modes_max, 
                       nmodes_to_report=None, initial_modes_dict={}, t0=0.,
                       t_ref=0., T=90., n_max=7, interpolate=True, use_news_power=True,
                       retrograde=False):
    """Calculates the fraction of unmodeled power and mismatch for each number
    of modes in nmodes_to_report. By default, the power is calculated using the News
    function. 

    Note that if retrograde is True, one (ell,m,n) mode is defined as including
    both prograde and retrograde solutions of (ell,m,n). If retrograde is
    False, then and (ell,m,n) mode counts as one mode.

    Parameters
    ----------
    W : WaveformModes object

    chif : float
        The dimensionless spin of the black hole, 0. <= chif < 1.

    M : float
        The mass of the black hole, M > 0.

    target_frac : float
        The target fractional power to leave unmodeled. Values should be 0. if
        one wants to model as much power as possible, in which case
        num_modes_max will be reached. If target_frac is non-zero and reached
        before num_modes_max, algorithm will exit with some modelled modes.

    num_modes_max : int
        Maximum number of modes to greedily chose.

    nmodes_to_report : list, optional [Default: None]
        List of modes to append on the way to num_modes_max.

    initial_modes_dict : dictionary, optional [Default: {}]
        Dictionary keys are tuples (l,m,n,sign) and values are complex amplitudes.

    t0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t0.

    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    T : float, optional [Default: 90.]
        The duration of the mismatch calculation, such that the end time of the
        mismatch integral is t0 + T.

    n_max : int, optional [Default: 7]
        Maximum overtone number to allow for a given mode (includes n_max).

    retrograde: boolean, optional [Default: False]
        All retrograde QNMs included if True, else only prograde mode is added
        except for m=0, where both are always added.

    Returns
    -------
    mode_dict : dict from fit_W_modes

    Q : WaveformModes object
      QNM model waveform corresponding to res_mode_dict

    diff : WaveformModes object
      Difference waveform W-Q

    frac_unmodeled_powers : list
      Fraction of power still unmodeled for each nmode_to_report

    wf_mismatches : list
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

    # Make sure that the waveform starts at t0, otherwise
    # inner_product returns faulty values due to interpolation issues
    if interpolate:
        W = W[np.argmin(abs(W.t - t0)):]
    
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
    W_power = np.real(W_power_waveform.inner_product(W_power_waveform, t1=t0))
        
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
        loudest_lms = mode_power_order(diff, topN=num_modes, t0=t0)
            
        mode_dict = add_modes(mode_dict, loudest_lms, n_max, retrograde)

        # Build a ringdown model
        qnms = list(mode_dict.keys())
        mode_dict = fit(W, chif, M, qnms, spherical_modes=None, t0=t0, t_ref=t_ref)

        Q = qnm_modes_as(chif, M, mode_dict, W, t0=t0, t_ref=t_ref)

        # How much power is unmodeled?
        if use_news_power:
            if W.dataType == scri.h:
                np.subtract(W.data, Q.data, out=diff.data)
                diff.data = diff.data_dot
            elif W.dataType == scri.hdot:
                np.subtract(W.data, Q.data, out=diff.data)
        else:
            np.subtract(W.data, Q.data, out=diff.data)
        diff_power = np.real(diff.inner_product(diff, t1=t0))
        frac_unmodeled_power = diff_power / W_power

        if i_mode+1 in nmodes_to_report:
            frac_unmodeled_powers.append(frac_unmodeled_power)
            wf_mismatches.append(mismatch(W, Q, t0, T,
                                                   spherical_modes=None))
            mode_dicts.append(mode_dict)

        if (frac_unmodeled_power < target_frac):
            break # Don't need to add any more modes

    frac_unmodeled_powers.append(frac_unmodeled_power)
    wf_mismatches.append(mismatch(W, Q, t0, T, spherical_modes=None))
    mode_dicts.append(mode_dict)
    # We've hit the max number of modes
    return mode_dicts, Q, diff, frac_unmodeled_powers, wf_mismatches

def pick_modes_greedy(W, chif, M, target_frac, num_modes_max, 
                      initial_modes_dict={}, t0=0., t_ref=0., T=90., n_max=7, 
                      interpolate=True, use_news_power=True, retrograde=False):
    """Calculates the fraction of unmodeled power and mismatch for the
    num_modes_max number of modes.
    """
    mode_dict, Q, diff, power, mismatch = pick_nmodes_greedy(
        W, 
        chif, 
        M, 
        target_frac,
        num_modes_max, 
        nmodes_to_report=None,
        initial_modes_dict=initial_modes_dict,
        t0=t0, 
        t_ref=t_ref,
        T=T,
        n_max=n_max,
        interpolate=interpolate,
        use_news_power=use_news_power,
        retrograde=retrograde)
    
    return mode_dict[0], Q, diff, power[0], mismatch[0]
