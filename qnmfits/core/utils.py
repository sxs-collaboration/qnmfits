import numpy as np
import jax.numpy as jnp

from jax.scipy.linalg import cholesky, solve_triangular

def mismatch(wf_array_1, wf_array_2, noise_covariance_matrix=None):
    """
    Compute the phase-maximized mismatch between two waveforms.

    If inv_noise_covariance_matrix is provided, the mismatch is computed 
    using the inverse noise covariance matrix.

    Parameters
    ----------
    wf_array_1 : array_like
        The first waveform array.
    wf_array_2 : array_like
        The second waveform array.
    inv_noise_covariance_matrix : array_like, optional
        The inverse noise covariance matrix. If `None`, the mismatch is
        computed without it.

    Returns
    -------
    float
        The mismatch value between the two waveforms.
    """

    if noise_covariance_matrix is None:

        numerator = np.real(
            np.sum(wf_array_1 * np.conj(wf_array_2))  # , axis=(0, 1))
        )
        wf_1_norm = np.real(
            np.sum(wf_array_1 * np.conj(wf_array_1))  #, axis=(0, 1))
        )
        wf_2_norm = np.real(
            np.sum(wf_array_2 * np.conj(wf_array_2))  #, axis=(0, 1))
        )

    else:

        L = cholesky(noise_covariance_matrix, lower=True)

        L_wf1 = solve_triangular(L, wf_array_1, lower=True)
        L_wf2 = solve_triangular(L, wf_array_2, lower=True)

        Kinv_wf1 = solve_triangular(
            jnp.transpose(L, [0, 2, 1]), L_wf1, lower=False
        )
        Kinv_wf2 = solve_triangular(
            jnp.transpose(L, [0, 2, 1]), L_wf2, lower=False
        )

        numerator = jnp.real(
            jnp.einsum("st, st -> ", jnp.conj(wf_array_1), Kinv_wf2).item()
        )
        wf_1_norm = jnp.real(
            jnp.einsum("st, st -> ", jnp.conj(wf_array_1), Kinv_wf1).item()
        )
        wf_2_norm = jnp.real(
            jnp.einsum("st, st -> ", jnp.conj(wf_array_2), Kinv_wf2).item()
        )

    denominator = np.sqrt(wf_1_norm * wf_2_norm)

    return 1 - (numerator / denominator)