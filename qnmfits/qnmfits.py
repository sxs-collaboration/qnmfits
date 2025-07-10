import numpy as np
import spherical_functions as sf

import scri
import bisect

from scipy.optimize import minimize

from .read_qnms import qnm_loader
qnm_loader = qnm_loader()


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


def window(data, t0, t0_method, T):
    """
    Window the data to the specified time range.

    Parameters
    ----------
    data : WaveformModes
        The waveform data to window.

    t0 : float
        The start time of the window.

    t0_method : str
        The method to use for determining the start time. Can be 'geq' or
        'closest'.

    T : float
        The duration of the window.

    Returns
    -------
    h_cut : WaveformModes
        The windowed waveform data.
    """

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

    return h_cut


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
        omega, C, C_ells = qnm_loader.qnm_from_tuple(
            (ell, m, n, sign), chif, Mf
        )

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


def fit(data, chif, Mf, t0, qnms, spherical_modes=None, T=100, t_ref=None,
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
    h_cut = window(data, t0=t0, t0_method=t0_method, T=T)

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
            omega, C, C_ells = qnm_loader.qnm_from_tuple(label, chif, Mf)

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


def fit_mass_spin(data, qnms, spherical_modes=None, t0=0, T=100, t_ref=None,
                  t0_method='geq', chif0=0.7, Mf0=0.95,
                  min_method='Nelder-Mead', min_options=None):
    """
    For a given set of QNMs and ringdown start time, find the remnant mass and
    spin which minimizes the mismatch between the ringdown model and data
    (calculated over the specified spherical-harmonic modes). This minimization
    is done via scipy.optimize.minimize, with the requested method. The complex
    QNM amplitudes are obtained via a least-squares fit, via qnmfits.fit.

    Parameters
    ----------
    data : WaveformModes
        Waveform data to fir the ringdown to. If only a subset of
        spherical-harmonic modes are to be used, specify this via the
        spherical_modes argument.

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
        the data time array (the same is true for the end time of the
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

    chif0 : float, optional [Default: 0.7]
        Initial guess for the best-fit remnant spin.

    Mf0 : float, optional [Default: 0.95]
        Initial guess for the best-fit remnant mass.

    min_method : str, optional [Default: 'Nelder-Mead']
        The method used to find the mismatch minimum in the mass-spin space.
        This can be any method available to scipy.optimize.minimize. This
        includes None, in which case the method is automatically chosen.

    min_options : dict, optional [Default: None]
        Dictionary passed to the options argument of scipy.optimize.minimize.
        If None, the following defaults are used:
        {'xatol': 1e-6, 'disp': False}.

    Returns
    -------
    result : dict
        A dictionary of useful information related to the fit. The keys are
        the same as for qnmfits.fit, with the addition of 'spin' and 'mass'.
    """
    # Initial guess for the minimization
    x0 = [chif0, Mf0]

    # Default options
    default_min_options = {'xatol': 1e-6, 'disp': False}

    # Merge user-provided options with defaults
    if min_options is not None:
        # User options override defaults
        default_min_options.update(min_options)

    bounds = [(0, 1.5), (0, 0.99)]

    # Wrapper for the qnmfits.fit function which returns the mismatch of the
    # fit, which we pass to scipy.optimize.minimize.
    def goodness(x, data, qnms, spherical_modes, t0, T, t_ref, t0_method):

        chif = x[0]
        Mf = x[1]

        if chif > 0.99:
            chif = 0.99
        if chif < 0:
            chif = 0

        best_fit = fit(
            data=data,
            chif=chif,
            Mf=Mf,
            qnms=qnms,
            spherical_modes=spherical_modes,
            t0=t0,
            T=T,
            t_ref=t_ref,
            t0_method=t0_method
        )

        return best_fit['mismatch']

    # Perform the SciPy minimization
    res = minimize(
        goodness,
        x0,
        args=(data, qnms, spherical_modes, t0, T, t_ref, t0_method),
        method=min_method,
        bounds=bounds,
        options=min_options
    )

    # The remnant properties that give the minimum mismatch
    chif_bestfit = res.x[0]
    Mf_bestfit = res.x[1]

    result = fit(
        data=data,
        chif=chif_bestfit,
        Mf=Mf_bestfit,
        qnms=qnms,
        spherical_modes=spherical_modes,
        t0=t0,
        T=T,
        t_ref=t_ref,
        t0_method=t0_method
    )

    result['spin'] = chif_bestfit
    result['mass'] = Mf_bestfit

    return result


# Greedy-fit functions
# --------------------

def mode_power_order(W, topN=10, t0=-np.inf):
    """Returns a list of topN indices sorted by power per mode for a waveform.

    Parameters
    ----------
    W : WaveformModes object

    topN : int, optional [Default: 10]

    t0 : float, optional [Default: -np.inf]
        Only compute power after t0.

    Returns
    -------
    mode_list : ndarray
      List of (l,m) indices, where the 0th row is the mode with the most
      power.
    """
    sliced = W.data[W.t > t0, :]
    return W.LM[
        np.argsort(np.sum(np.square(np.abs(sliced)), axis=0))
    ][-1:-1-topN:-1]


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
         Dictionary keys are tuples (l,m,n,sign) and values are complex
         amplitudes.

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
    # Loop through loudest_lms till we find a mode we can add
    for loudest_l, loudest_m in loudest_lms:
        ns_so_far = list({n for (l, m, n, _) in modes_so_far_dict.keys()
                          if ((l == loudest_l) and (m == loudest_m))})

        max_n_so_far = np.max(ns_so_far) if len(ns_so_far) > 0 else -1
        new_n = max_n_so_far + 1

        if new_n <= n_max:
            if (loudest_m == 0) or retrograde:
                new_modes[(loudest_l, loudest_m, new_n, +1)] = None
                new_modes[(loudest_l, loudest_m, new_n, -1)] = None
            else:
                new_modes[(loudest_l, loudest_m, new_n,
                           int(np.sign(loudest_m)))] = None
            break
        elif (
            new_n > n_max and
            (loudest_l, loudest_m) == (loudest_lms[-1][0], loudest_lms[-1][1])
        ):
            print("Cannot find a valid mode to add.")
    return new_modes


def pick_nmodes_greedy(data, chif, Mf, t0, target_frac, num_modes_max,
                       initial_modes_dict={}, T=100, t_ref=None,
                       t0_method='geq', n_max=7, use_news_power=True,
                       retrograde=False):
    """
    Calculates the fraction of unmodeled power and mismatch for each number
    of modes in nmodes_to_report. By default, the power is calculated using the
    News function.

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
        Dictionary keys are tuples (l,m,n,sign) and values are complex
        amplitudes.

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
    if t_ref is None:
        t_ref = t0

    # Make sure that target_frac is between 0 and 1
    if not 0 <= target_frac <= 1:
        raise ValueError(
            f"target_frac={target_frac} should be between 0 and 1."
        )

    # Make sure that num_modes_max is a non-negative integer
    if not num_modes_max > 0:
        raise ValueError(
            f"num_modes_max={num_modes_max} must be greater than 0."
        )

    if None in initial_modes_dict.values():
        raise ValueError(
            "All values in initial_modes_dict should be complex numbers."
        )

    # if nmodes_to_report is None:
    #     nmodes_to_report = []

    # Window the data
    data = window(data, t0=t0, t0_method=t0_method, T=T)

    if use_news_power:
        if data.dataType == scri.h:
            W_power_waveform = data.copy()
            W_power_waveform.data = data.data_dot
            W_power_waveform.dataType = scri.hdot
        elif data.dataType == scri.hdot:
            W_power_waveform = data.copy()
        else:
            raise ValueError("W is not of type scri.h or scri.hdot")
    else:
        W_power_waveform = data.copy()

    W_power = np.real(
        W_power_waveform.inner_product(W_power_waveform, t1=t0, t2=t0+T)
    )

    # Initially, the difference between the waveform and model is just the
    # waveform itself
    diff = W_power_waveform.copy()

    mode_dict = initial_modes_dict.copy()
    for i_mode in np.arange(0, num_modes_max):

        # Add one or two modes
        num_modes = len(data.LM)
        loudest_lms = mode_power_order(diff, topN=num_modes, t0=t0)
        mode_dict = add_modes(mode_dict, loudest_lms, n_max, retrograde)

        # Build a ringdown model
        qnms = list(mode_dict.keys())
        best_fit = fit(
            data=data,
            chif=chif,
            Mf=Mf,
            t0=t0,
            qnms=qnms,
            spherical_modes=None,
            t_ref=t_ref
        )
        amp_dict = best_fit['amplitudes']

        Q = qnm_WaveformModes(
            times=data.t,
            chif=chif,
            Mf=Mf,
            qnm_amps=amp_dict,
            t0=t0,
            t_ref=t_ref,
            ell_min=data.ell_min,
            ell_max=data.ell_max,
            t0_method=t0_method,
        )

        # How much power is unmodeled?
        if use_news_power:
            if data.dataType == scri.h:
                np.subtract(data.data, Q.data, out=diff.data)
                diff.data = diff.data_dot
            elif data.dataType == scri.hdot:
                np.subtract(data.data, Q.data, out=diff.data)
        else:
            np.subtract(data.data, Q.data, out=diff.data)
        diff_power = np.real(diff.inner_product(diff, t1=t0, t2=t0+T))
        frac_unmodeled_power = diff_power / W_power

        # if i_mode+1 in nmodes_to_report:
        #     frac_unmodeled_powers.append(frac_unmodeled_power)
        #     wf_mismatches.append(
        #         mismatch(h_A=data, h_B=Q, t0=t0, T=T, spherical_modes=None)
        #     )
        #     mode_dicts.append(mode_dict)

        if frac_unmodeled_power < target_frac:
            break  # Don't need to add any more modes

    wf_mismatch = mismatch(h_A=data, h_B=Q, t0=t0, T=T, spherical_modes=None)

    # We've hit the max number of modes
    return amp_dict, Q, diff, frac_unmodeled_power, wf_mismatch
