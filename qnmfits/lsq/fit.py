import numpy as np

from qnmfits.core.utils import mismatch

def fit(times, data, frequencies, t0, T=100, t_ref=None, t0_method="geq"):
    """
    The base single-mode ringdown fitting function. This finds the best-fit (as
    determined by a least-squares fit) amplitudes and phases of the sum of
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

    Returns
    -------
    result : dict
        A dictionary of useful information related to the fit. See below for
        details.

    Notes
    -----
    The output dictionary contains the following entries:
    'residual' : float
        The sum of residuals from the least-squares fit.
    'rank' : int
        The rank of the coefficient matrix used in the least-squares fit.
    's' : array_like
        The singular values of the coefficient matrix used in the least-squares
        fit.
    'mismatch' : float
        The mismatch between the data and the best-fit model over the fitting
        window.
    'C' : array_like
        The complex amplitudes of the best-fit quasinormal modes. This has
        shape (len(frequencies),), and is ordered corresponding to the input
        frequencies.
    'times' : array_like
        The times used in the fit (after any masking).
    'data' : array_like
        The data used in the fit (after any masking).
    'model' : array_like
        The best-fit model evaluated at the times used in the fit.
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
        
    # Construct the coefficient matrix. This has shape (len(times_masked),
    # len(frequencies)). Each column corresponds to a mode, and each row to a
    # time.
    a = np.array(
        [
            np.exp(-1j*frequencies[i]*(times_masked-t0)) 
            for i in range(len(frequencies))
        ]
    ).T

    # Solve for the complex amplitudes, C. Also returns the sum of residuals,
    # the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data_masked, rcond=None)

    # Evaluate the model
    model = np.einsum("ij,j->i", a, C)

    # Calculate the mismatch for the fit
    mm = mismatch(model, data_masked)

    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'rank': rank,
        's': s,
        'mismatch': mm,
        'C': C,
        'data': data_masked,
        'model': model,
        'times': times_masked,
        't0': t0,
        'frequencies': frequencies
        }

    # Return the output dictionary
    return best_fit
