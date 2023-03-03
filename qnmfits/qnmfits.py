import numpy as np

# Class to load QNM frequencies and mixing coefficients
from .qnm import qnm
qnm = qnm()


def mismatch(times, wf_1, wf_2):
    """
    Calculates the mismatch between two complex waveforms.
    
    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.
        
    wf_1, wf_2 : array_like
        The two waveforms to calculate the mismatch between.
        
    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """
    numerator = np.real(np.trapz(wf_1 * np.conjugate(wf_2), x=times))
    
    denominator = np.sqrt(
        np.trapz(np.real(wf_1 * np.conjugate(wf_1)), x=times)
        *np.trapz(np.real(wf_2 * np.conjugate(wf_2)), x=times)
        )
    
    return 1 - (numerator/denominator)


def multimode_mismatch(times, wf_dict_1, wf_dict_2):
    """
    Calculates the multimode (sky-averaged) mismatch between two dictionaries 
    of spherical-harmonic waveform modes. 
    
    If the two dictionaries have a different set of keys, the sum is performed
    over the keys of wf_dict_1 (this may be the case, for example, if only a 
    subset of spherical-harmonic modes are modelled).
    
    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.
        
    wf_dict_1, wf_dict_2 : dict
        The two dictionaries of waveform modes to calculate the mismatch 
        between.
        
    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """    
    keys = list(wf_dict_1.keys())
    
    numerator = np.real(sum([
        np.trapz(wf_dict_1[key] * np.conjugate(wf_dict_2[key]), x=times) 
        for key in keys]))
    
    wf_1_norm = sum([
        np.trapz(np.real(wf_dict_1[key] * np.conjugate(wf_dict_1[key])), x=times) 
        for key in keys])
    
    wf_2_norm = sum([
        np.trapz(np.real(wf_dict_2[key] * np.conjugate(wf_dict_2[key])), x=times) 
        for key in keys])
    
    denominator = np.sqrt(wf_1_norm*wf_2_norm)
    
    return 1 - (numerator/denominator)


def ringdown_fit(data, spherical_mode, qnms, Mf, chif, t0, t0_method='geq', T=100):
    """
    Perform a least-squares fit to some data using a ringdown model.
    
    Parameters
    ----------
    data : WaveformModes
        The data to be fitted by the ringdown model.
    
    spherical_mode : tuple
        The (l,m) mode to fit with the ringdown model.
        
    qnms : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float
        The magnitude of the remnant black hole spin.
        
    t0 : float
        The start time of the ringdown model.
        
    t0_method : str, optional
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
                
        The default is 'geq'.
        
    T : float, optional
        The duration of the data to analyse, such that the end time is t0 + T. 
        The default is 100.
        
    Returns
    -------
    best_fit : dict
        A dictionary of useful information related to the fit. Keys include:
            
            - 'residual' : float
                The residual from the fit.
            - 'mismatch' : float
                The mismatch between the best-fit waveform and the data.
            - 'C' : ndarray
                The best-fit complex amplitudes. There is a complex amplitude 
                for each ringdown mode.
            - 'frequencies' : ndarray
                The values of the complex frequencies for all the ringdown 
                modes.
            - 'data' : ndarray
                The (masked) data used in the fit.
            - 'model': ndarray
                The best-fit model waveform.
            - 'times' : ndarray
                The times at which the data and model are evaluated.
            - 't0' : float
                The ringdown start time used in the fit.
            - 'modes' : ndarray
                The ringdown modes used in the fit.
    """
    # Get the data array we want to fit to
    times = data.t
    data = data.data[:, data.index(*spherical_mode)]
    
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = data[data_mask]
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        
        times = times[start_index:end_index]
        data = data[start_index:end_index]
        
    else:
        print("""Requested t0_method is not valid. Please choose between 'geq'
              and 'closest'""")
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(qnms, chif, Mf))
        
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.array([
        np.exp(-1j*frequencies[i]*(times-t0)) for i in range(len(frequencies))
        ]).T

    # Solve for the complex amplitudes, C. Also returns the sum of residuals,
    # the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model
    model = np.einsum('ij,j->i', a, C)
    
    # Calculate the mismatch for the fit
    mm = mismatch(times, model, data)
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'frequencies': frequencies,
        'data': data,
        'model': model,
        'times': times,
        't0': t0,
        }
    
    # Return the output dictionary
    return best_fit

    
def multimode_ringdown_fit(data, spherical_modes, qnms, Mf, chif, t0, 
                           t0_method='geq', T=100):
    
    # Extract the times from the WaveformModes object
    times = data.t
    
    # Index the requested spherical modes
    data = data.data[
        :, [data.index(*spherical_mode) for spherical_mode in spherical_modes]
        ]
    
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = data[data_mask]
        
    elif t0_method == 'closest':
        
        # Use data.index_closest_to?
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        
        times = times[start_index:end_index]
        data = data[start_index:end_index,:]
        
    else:
        print("""Requested t0_method is not valid. Please choose between 'geq'
              and 'closest'.""")
              
    # The data in the form of a dictionary will be useful for the mismatch
    # calculation. In future turn everything into WaveformModes?
    data_dict = {mode: data[:,i] for i, mode in enumerate(spherical_modes)}
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(qnms, chif, Mf))
    
    # Construct the coefficient matrix for use with NumPy's lstsq function. 
    
    # Mixing coefficients
    # -------------------
    
    # A list of lists for the mixing coefficient indices. The first list is
    # associated with the first spherical mode. The second list is associated 
    # with the second spherical mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
    indices_lists = [
        [spherical_mode+qnm for qnm in qnms] for spherical_mode in spherical_modes
        ]
    
    # Convert each tuple of indices in indices_lists to a mu value
    mu_lists = np.array([qnm.mu_list(indices, chif) for indices in indices_lists])
    
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    a = np.vstack(
        [mu_list*np.exp(-1j*np.outer(times-t0, frequencies)) for mu_list in mu_lists]
        )

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, np.hstack(data.T), rcond=None)
    
    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum('ij,j->i', a, C)
    
    # Split up the result into the separate spherical harmonic modes, and
    # store to a dictionary. We also store the "weighted" complex amplitudes 
    # to a dictionary.
    model_dict = {}
    weighted_C = {}
    
    for i, lm in enumerate(spherical_modes):
        model_dict[lm] = model[i*len(times):(i+1)*len(times)]
        weighted_C[lm] = np.array(mu_lists[i])*C
    
    # Calculate the (sky-averaged) mismatch for the fit
    mm = multimode_mismatch(times, model_dict, data_dict)
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'weighted_C': weighted_C,
        'frequencies': frequencies,
        'data': data_dict,
        'model': model_dict,
        'times': times,
        't0': t0,
        }
    
    # Return the output dictionary
    return best_fit