import numpy as np
from qnmfits.bgp import base
from qnmfits.core.utils import mismatch

def fit(times, data, frequencies, t0, T=100, t_ref=None, t0_method="geq", K = 1, N_samples=10000):
    """
    The base single-mode ringdown fitting function. This finds the best-fit (as
    determined by a Bayesian fit which, for a white noise covariance is equivalent
    to a least-squares fit) amplitudes and phases of the sum of
    quasinormal-modes with the specified frequencies to the provided data.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
    data : array_like
        The data to be fitted by the ringdown model.
    frequencies : list of complex
        List of quasinormal mode frequencies to fit.
    t0 : float
        Start time for the fitting window.
    T : float, optional
        Duration of the fitting window (default is 100).
    t_ref : float, optional
        Reference time for the waveform data (default is None).
    t0_method : {"geq", "closest"}, optional
        Method to window the data (default is "geq"). A requested ringdown
        start time will in general lie between times on the default time array
        (the same is true for the end time of the analysis). There are
        different approaches to deal with this, which can be specified here.
        Options are: "geq" (take data at times greater than or equal to t0),
        "closest" (identify the data point occurring at a time closest to t0,
        and take times from there).
    K : int or array_like, optional
        A noise covariance specification. If an integer is provided, this is
        interpreted as the inverse variance of white noise. If an array is
        provided, this is interpreted as the full noise covariance matrix.
        Default is 1 (corresponding to the base least-squares case). 
    N_samples : int, optional
        The number of samples to draw from the posterior distribution. Default is 10000.

    Returns
    -------
    result : dict
        A dictionary of useful information related to the fit. See below for
        details.

    Notes
    -----
    The output dictionary contains the following entries:
    'samples' : array_like
        Samples drawn from the posterior distribution of the fit parameters.
    'mean_vector' : array_like
        The mean vector of the posterior distribution.
    'covariance_matrix' : array_like
        The covariance matrix of the posterior distribution.
    'times' : array_like
        The times used in the fit (after any masking).
    'data' : array_like
        The data used in the fit (after any masking).
    't0' : float
        The requested start time of the fitting window.
    'frequencies' : list of complex
        The quasinormal-mode frequencies used in the fit.
    """
    # If no reference time is provided, set it to the start time of the fit
    if t_ref is None:
        t_ref = t0

    # Mask the data with the requested method
    if t0_method == "geq":
        data_mask = (times >= t0) & (times < t0 + T)
        times_masked = times[data_mask]
        data_masked = data[data_mask]

    elif t0_method == "closest":
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        times_masked = times[start_index:end_index]
        data_masked = data[start_index:end_index]

    else:
        print("""Requested t0_method is not valid. Please choose between "geq"
              and "closest".""")
        
    model_times = times_masked - t0

    a = base.get_exponential_terms(model_times, frequencies, len(frequencies)).T

    # Solve for the complex amplitudes, C. Also returns the sum of residuals,
    # the rank of a, and singular values of a.
    C, _, _, _ = np.linalg.lstsq(a, data_masked, rcond=None)
    ref_params = np.concatenate([np.stack([np.real(C), np.imag(C)], axis=-1).reshape(-1)])

    #Mixing is explicitly left in the functions to preserve the structure used for multimode fits
    mixing = np.ones((1, len(frequencies)))

    # Construct model matrix 
    re_model_terms = base.get_re_amplitude_term(a.T, mixing)
    model_terms = np.zeros((len(frequencies)*2, 1, len(model_times)),dtype=np.complex128)
    model_terms[::2], model_terms[1::2] = re_model_terms, 1j * re_model_terms
    model_terms = np.moveaxis(model_terms, 0, -1)

    const_term = base.get_const_term(C, a.T, mixing)

    is_GP_diagonal, L = base.prepare_noise_covariance(K, len(model_times))
    fisher_matrix = base.get_fisher_matrix(model_times, model_terms, L, is_GP_diagonal)
    b_vector = base.get_b_vector(data_masked, const_term, model_times, model_terms, L, is_GP_diagonal)
    mean_vector = np.linalg.solve(fisher_matrix, b_vector) + ref_params
    covariance_matrix = base.get_inverse(fisher_matrix)

    samples = base.get_samples(mean_vector, covariance_matrix, num_samples=N_samples)

    # Store all useful information to a output dictionary
    best_fit = {
        'samples': samples,
        'mean_vector': mean_vector,
        'covariance_matrix': covariance_matrix,
        'data': data_masked,
        'times': times_masked,
        't0': t0,
        'frequencies': frequencies
        }

    # Return the output dictionary
    return best_fit