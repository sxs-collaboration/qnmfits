import qnm
import numpy as np

from pathlib import Path
from urllib.request import urlretrieve
from scipy.interpolate import CubicSpline

import h5py

_ksc = qnm.modes_cache


def download_cook_data():
    """
    Download data for the n=8 and n=9 QNMs from
    https://zenodo.org/records/10093311, and store in the qnmfits/Data
    directory.
    """

    data_dir = Path(__file__).parent / 'Data'
    data_dir.mkdir(parents=True, exist_ok=True)

    for n in [8, 9]:

        download_url = (
            'https://zenodo.org/records/10093311/files/'
            f'KerrQNM_{n:02}.h5?download=1'
        )
        file_path = data_dir / f'KerrQNM_{n:02}.h5'

        if not file_path.exists():
            print(f'Downloading KerrQNM_{n:02}.h5...')
            urlretrieve(download_url, file_path)
        else:
            print(f'KerrQNM_{n:02}.h5 already downloaded.')


class qnm_loader:
    """
    """

    def __init__(self):
        """
        Initialize the class.
        """

        # The method used by the qnm package breaks down for certain modes that
        # approach the imaginary axis (perhaps most notably, the (2,2,8) mode).
        # We load data for these modes separately, computed by Cook &
        # Zalutskiy.

        data_dir = Path(__file__).parent / 'Data'

        # Dictionary to store the mode data, using our preferred labeling
        # convention
        multiplet_data = {}

        self._multiplet_funcs = {}

        # A list of multiplets (this list is not complete!)
        self.multiplet_list = [(2, 0, 8), (2, 1, 8), (2, 2, 8)]

        # Keep track of what data has been downloaded (useful for warnings)
        self.download_check = {}

        for ell, m, n in self.multiplet_list:

            file_path = data_dir / f'KerrQNM_{n:02}.h5'
            self.download_check[n] = file_path.exists()
            if self.download_check[n]:

                # Open file
                with h5py.File(file_path, 'r') as f:

                    # Read data for each multiplet, and store in the
                    # multiplet_data dictionary with the preferred labelling
                    # convention
                    for i in [0, 1]:
                        multiplet_data[(ell, m, n+i)] = np.array(
                            f[f'n{n:02}/m{m:+03}/{{{ell},{m},{{{n},{i}}}}}']
                            )

        for key, data in multiplet_data.items():

            # Extract relevant quantities
            spins = data[:, 0]
            omega = data[:, 1] + 1j*data[:, 2]
            all_C = data[:, 5::2] + 1j*data[:, 6::2]

            # Interpolate omegas
            omega_interp = CubicSpline(spins, omega)

            # Interpolate Cs
            all_C_interp = []
            for C in all_C.T:
                all_C_interp.append(CubicSpline(spins, C))

            # Add these interpolated functions to the _multiplet_funcs
            # dictionary
            self._multiplet_funcs[key] = [omega_interp, all_C_interp]

    def load_qnm(self, ell, m, n, chi, s=-2):

        if (ell, m, n) in self._multiplet_funcs:
            omega_func, all_C_funcs = self._multiplet_funcs[ell, m, n]
            omega = omega_func(chi)
            all_C = np.array([C_func(chi) for C_func in all_C_funcs])
            ell_max = len(all_C) + ell - 1

        else:

            # If there is a known multiplet with the same ell and m, we need to
            # be careful with the n index
            n_load = n
            for ellp, mp, nprime in self.multiplet_list:
                if (ell == ellp) & (m == mp):
                    if n > nprime+1:
                        n_load -= 1

            mode_seq = _ksc(s, ell, m, n_load)
            omega, _, all_C = mode_seq(chi, store=True)
            ell_max = mode_seq.l_max

        ells = qnm.angular.ells(s, m, ell_max)

        return omega, all_C, ells

    def qnm_from_tuple(self, tup, chi, M=1, s=-2):

        ell, m, n, sign = tup

        if (sign == +1):
            omega, all_C, ells = self.load_qnm(ell, m, n, chi, s)
        elif (sign == -1):
            omega, all_C, ells = self.load_qnm(ell, -m, n, chi, s)

        if (sign == -1):
            omega = -np.conj(omega)
            all_C = (-1)**(ell + ells) * np.conj(all_C)

        omega /= M

        return omega, all_C, ells


# def qnm_from_tuple(tup, chi, M, s=-2):
#     """
#     Get QNM frequency and spherical-spheroidal mixing coefficients from the
#     qnm package.

#     Parameters
#     ----------
#     tup : tuple
#         Index (ell,m,n,sign) of QNM

#     chi : float
#         The dimensionless spin of the black hole, 0. <= chi < 1.

#     M : float
#         The mass of the final black hole, M > 0.

#     s : int, optional [Default: -2]

#     Returns
#     -------
#     omega: complex
#         Frequency of QNM. This frequency is the same units as arguments,
#         as opposed to being in units of remnant mass.

#     C : complex ndarray
#         Spherical-spheroidal decomposition coefficient array

#     ells : ndarray
#         List of ell values for the spherical-spheroidal mixing array

#     """
#     ell, m, n, sign = tup

#     # Use separate data for this special mode. The QNM frequency and angular
#     # separation constants are provided.
#     if (ell, m, n) == (2, 2, 8):

#         w228table = np.loadtxt('../qnmfits/data/w228table.dat')
#         spins, real_omega, imag_omega, real_A, imag_A = w228table.T
#         omega = real_omega + 1j*imag_omega

#         CS = CubicSpline(spins, omega)
#         omega = CS(chi)

#         if (sign == -1):
#             omega = -np.conj(omega)

#         return omega, None, None

#     else:
#         if (sign == +1):
#             mode_seq = _ksc(s, ell, m, n)
#         elif (sign == -1):
#             mode_seq = _ksc(s, ell, -m, n)
#         else:
#             raise ValueError(
#                 'Last element of mode label must be +1 or -1, '
#                 f'instead got {sign}'
#             )

#         # The output from mode_seq is M*omega
#         try:
#             Momega, _, C = mode_seq(chi, store=True)
#         except:
#             Momega, _, C = mode_seq(chi, interp_only=True)

#         ells = qnm.angular.ells(s, m, 16)

#         if (sign == -1):
#             Momega = -np.conj(Momega)
#             C = (-1)**(ell + ells) * np.conj(C)

#         # Convert from M*\omega to \omega
#         omega = Momega/M
#         return omega, C, ells
