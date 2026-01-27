import numpy as np
from qnmfits.read_qnms import qnm_loader
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.linalg import cholesky 

qnmloader = qnm_loader() 

def get_frequency(mode, chif, Mf):
    """Compute the frequency for a given QNM mode."""
    omega, _, _ = qnmloader.qnm_from_tuple(mode, chif, Mf)
    return omega

def get_domega_dchif(mode, chif, Mf, delta=1.0e-6):
    """Compute the derivative of the frequency with respect to chif for a given QNM mode."""
    omega_plus = get_frequency(mode, chif + delta, Mf)
    omega_minus = get_frequency(mode, chif - delta, Mf)
    return (omega_plus - omega_minus) / (2 * delta)

def get_mixing(mode, sph_mode, chif, Mf):
    """Compute the mixing coefficient for a given QNM mode and spherical mode."""
    l = sph_mode[0] 
    _, mu, _ = qnmloader.qnm_from_tuple(mode, chif, Mf)
    return mu[l - 2]

def get_dmu_dchif(mode, sph_mode, chif, Mf, delta=1.0e-6):
    """Compute the derivative of the mixing coefficient with respect to chif."""
    mu_plus = get_mixing(mode, sph_mode, chif=chif + delta, Mf=Mf)
    mu_minus = get_mixing(mode, sph_mode, chif=chif - delta, Mf=Mf)
    return (mu_plus - mu_minus) / (2 * delta)

def get_exponential_terms(analysis_times, frequencies, modes_length):
    """Compute the exponential terms for the QNM modes at the given analysis times."""
    return np.array([np.exp(-1j * frequencies[i] * analysis_times) for i in range(modes_length)])

def get_re_amplitude_term(exponential_terms, mixing_coefficients):
    """Computes the model term for the real part of the amplitudes for a given set of QNMs."""
    return np.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

def get_im_amplitude_term(exponential_terms, mixing_coefficients):
    """Computes the model term for the imaginary part of the amplitudes for a given set of QNMs."""
    return np.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

def get_mass_term(analysis_times, ls_amplitudes, frequencies, exponential_terms, mixing_coefficients, Mf):
    """Computes the mass term for a given set of QNMs."""
    return np.einsum(
        "p,sp,pt->st",
        1j * frequencies / Mf * ls_amplitudes,
        mixing_coefficients,
        exponential_terms * analysis_times,
    )

def get_chif_term(analysis_times, ls_amplitudes, exponential_terms, mixing_coefficients, mixing_derivatives, frequency_derivatives):
    """Computes the chif term for a given set of QNMs."""
    return np.einsum(
        "p,pt,spt->st",
        ls_amplitudes,
        exponential_terms,
        mixing_derivatives[:, :, None]
        - 1j * mixing_coefficients[:, :, None] * analysis_times[None, None, :]
        * frequency_derivatives[None, :, None],
    )

def get_const_term(ls_amplitudes, exponential_terms, mixing_coefficients):
    """Computes the constant (reference) term for the model."""
    return np.einsum("p,sp,pt->st", ls_amplitudes, mixing_coefficients, exponential_terms)

def prepare_noise_covariance(K, len_times):
    """Prepare the noise covariance matrix and its lower triangular decomposition."""
    if isinstance(K, (int, float)):
        is_GP_diagonal = True
        noise_covariance_matrix = np.eye(len_times) * K
    else:
        is_GP_diagonal = False
        noise_covariance_matrix = K

    noise_covariance_lower_triangular = cholesky(noise_covariance_matrix, lower=True)
    return is_GP_diagonal, noise_covariance_lower_triangular[np.newaxis, ...]

def get_fisher_matrix(analysis_times, model_terms, noise_covariance_lower_triangular, is_GP_diagonal):
    """Computes the Fisher matrix for the parameters.
    
    This method uses the Cholesky decomposition of the noise covariance matrix 
    to compute the Fisher matrix efficiently without direct inversion.

    Args:
        analysis_times (np.ndarray): The times at which the Fisher matrix is evaluated.
        model_terms (np.ndarray): A 3D array of model terms of shape: spherical modes x analysis times x model parameters.
        noise_covariance_lower_triangular (np.ndarray): The lower triangular Cholesky decomposition of the noise covariance matrix.
        is_GP_diagonal (bool): Whether the noise covariance is diagonal.
        
    Returns:
        np.ndarray: A 2D array representing the Fisher matrix.
    """
    L_model_terms = solve_triangular(noise_covariance_lower_triangular, model_terms, lower=True)
    Kinv_model_terms = solve_triangular(
        np.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_model_terms, lower=False
    )
    fisher_matrix = np.einsum("btm, btn -> mn", np.conj(model_terms), Kinv_model_terms)

    if is_GP_diagonal:
        return np.real(fisher_matrix) * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
    else:
        return np.real(fisher_matrix)

def get_b_vector(data_array, constant_term, analysis_times, model_terms, noise_covariance_lower_triangular, is_GP_diagonal):
    """
        Computes the b vector for the parameters in `params`. This method uses the Cholesky decomposition
        of the noise covariance matrix to compute the b vector efficiently and without direct inversion of the covariance matrix.

        Args:
            data_array (np.ndarray): The data array of shape: spherical modes x analysis times.
            constant_term (np.ndarray): The constant term computed with reference parameters.
            analysis_times (np.ndarray): The times at which the b vector is evaluated.
            model_terms (np.ndarray): A 3D array of model terms of shape: spherical modes x analysis times x model parameters.
            noise_covariance_lower_triangular (np.ndarray): The lower triangular Cholesky decomposition of the noise covariance matrix.
        Returns:
            np.ndarray: A 1D NumPy array representing the b vector, where each element corresponds to a parameter in `params`.
    """
    data_array_new = data_array - constant_term
    L_model_terms = solve_triangular(noise_covariance_lower_triangular, model_terms, lower=True)
    Kinv_model_terms = solve_triangular(
        np.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_model_terms, lower=False
    )
    b_vector = np.einsum("st, stp -> p", np.conj(data_array_new), Kinv_model_terms)

    if is_GP_diagonal:
        return np.real(b_vector) * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
    else:
        return np.real(b_vector)

def get_inverse(matrix, epsilon=1e-10):
    """
    Compute the inverse of a matrix using its eigenvalues and eigenvectors.

    Args:
        matrix (array): The matrix to be inverted.
        epsilon (float): A small value to ensure numerical stability.
    Returns:
        array: The inverse of the input matrix.

    """
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, epsilon)
    return np.einsum("ik, k, jk -> ij", vecs, 1 / vals, vecs)

def get_samples(mean_vector, covariance_matrix, num_samples):
    """
    Draw samples from a multivariate normal distribution with the given mean vector and covariance matrix.
    Args:
        mean_vector (array): The mean vector of the multivariate normal distribution.
        covariance_matrix (array): The covariance matrix of the multivariate normal distribution.
    Returns:
        samples (array): Samples drawn from the multivariate normal distribution.
    """
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=num_samples)
    return samples
