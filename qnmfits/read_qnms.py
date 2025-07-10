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
    Helper class to load quasinormal-mode frequencies and spherical-spheroidal
    mixing coefficients. We default to using the qnm package, but for some
    special modes we load separate data.
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

        # We will interpolate the data and store to this dictionary
        self._multiplet_funcs = {}

        # A list of multiplets (this list is not complete!)
        self.multiplet_list = [(2, 0, 8), (2, 1, 8), (2, 2, 8)]

        for ell, m, n in self.multiplet_list:

            file_path = data_dir / f'KerrQNM_{n:02}.h5'
            if file_path.exists():

                # Open file
                with h5py.File(file_path, 'r') as f:

                    # Read data for each multiplet, and store in the
                    # multiplet_data dictionary with the preferred labeling
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

    def load_qnm(self, ell, m, n, chif, s=-2):
        """
        Helper function to load a requested quasinormal-mode frequency and
        associated mixing coefficients. In practice, the qnm_from_tuple
        function will usually want to be used.

        Note that this function is designed to only return frequencies and
        mixing coefficients associated with the "regular" (positive real part)
        modes. Note also that the frequency is given in units of the remnant
        black-hole mass (Mf*omega).

        Parameters
        ----------
        ell : int
            The angular number of the mode.

        m : int
            The azimuthal number of the mode.

        n : int
            The overtone number of the mode.

        chif : float or array_like
            The dimensionless spin magnitude of the black hole.

        s : int, optional [Default: -2]
            The spin weight of the mode.

        Returns
        -------
        omega : complex
            The quasinormal-mode frequency, Mf*omega

        all_C : complex ndarray
            Spherical-spheroidal mixing coefficients for each (ell',m) up to
            some ell_max.

        ells: list
            The ells associated with the returned mixing coefficients.
        """

        if (ell, m, n) in self._multiplet_funcs:
            omega_func, all_C_funcs = self._multiplet_funcs[ell, m, n]
            omega = omega_func(chif)
            all_C = np.array([C_func(chif) for C_func in all_C_funcs])
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
            omega, _, all_C = mode_seq(chif, store=True)
            ell_max = mode_seq.l_max

        ells = qnm.angular.ells(s, m, ell_max)

        return omega, all_C, ells

    def qnm_from_tuple(self, tup, chif, Mf=1, s=-2):

        ell, m, n, sign = tup

        if (sign == +1):
            omega, all_C, ells = self.load_qnm(ell, m, n, chif, s)
        elif (sign == -1):
            omega, all_C, ells = self.load_qnm(ell, -m, n, chif, s)

        if (sign == -1):
            omega = -np.conj(omega)
            all_C = (-1)**(ell + ells) * np.conj(all_C)

        omega /= Mf

        return omega, all_C, ells
