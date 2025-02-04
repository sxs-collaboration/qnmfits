import numpy as np
import spherical_functions as sf

import os
import scri
import sxs
import pickle
import quaternion

from .cce import cce
cce = cce()


def dict_to_WaveformModes(times, data, ell_min=2, ell_max=None):
    """
    Converts a dictionary of data arrays to a scri WaveformModes object.

    Parameters
    ----------
    times : array
        The time array for the data.

    data : dictionary
        A dictionary of data arrays. The keys are (ell, m) tuples.

    ell_min : int, optional [Default: 2]
        The minimum ell value to include in the WaveformModes object.

    ell_max : int, optional [Default: None]
        The maximum ell value to include in the WaveformModes object. If not
        specified, the maximum ell value in the dictionary will be used.

    Returns
    -------
    wm : scri WaveformModes object
    """
    # If not specified, obtain ell_max from the dictionary keys
    if ell_max is None:
        ell_max = max([ell for ell, _ in data.keys()])

    # The spherical-harmonic mode (ell, m) indices for the requested ell_min
    # and ell_max
    ell_m_list = sf.LM_range(ell_min, ell_max)

    # Initialize the WaveformModes data array
    wm_data = np.zeros((len(times), len(ell_m_list)), dtype=complex)

    # Fill the WaveformModes data array
    for i, (ell, m) in enumerate(ell_m_list):
        if (ell, m) in data.keys():
            wm_data[:, i] = data[(ell, m)]

    # Construct the WaveformModes object
    wm = scri.WaveformModes(
        dataType=scri.h,
        t=times,
        data=wm_data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True
    )

    return wm


def sxs_to_scri_WaveformModes(wm_sxs):
    """
    Converts an sxs WaveformModes object to a scri WaceformModes object.

    Parameters
    ----------
    h_sxs : sxs WaveformModes object

    Returns
    -------
    h : scri WaveformModes object
    """

    wm = scri.WaveformModes(
        dataType=scri.h,
        t=wm_sxs.t,
        data=wm_sxs.data,
        ell_min=wm_sxs.ell_min,
        ell_max=wm_sxs.ell_max,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True
    )

    return wm


def to_superrest_frame(abd, t0, window=True):
    """
    Map an AsymptoticBondiData object to the superrest frame.

    Parameters
    ----------
    abd : AsymptoticBondiData
        The simulation data.

    t0 : float
        The time at which the superrest frame is defined. For ringdown
        studies, about 300M after the time of peak strain is recommended.

    window : bool, optional [Default: True]
        Whether to window the data to speed up the transformation. Waveform
        data 100M before the peak of the strain is removed.

    Returns
    -------
    abd_prime : AsymptoticBondiData
        The simulation data in the superrest frame.
    """
    # The extraction radius of the simulation
    R = int(abd.metadata['preferred_R'])

    # Check if the transformation to the superrest frame has already been
    # done
    wf_path = \
        abd.sim_dir / f'rhoverM_BondiCce_R{R:04d}_t0{t0}_superrest.pickle'

    if not wf_path.is_file():

        # Window the data to speed up the transformation
        if window:

            # Shift the zero time to be at the peak of the strain
            time_shift = abd.t[np.argmax(abd.h.norm())]
            abd.t -= time_shift

            # The scri interpolation removes the metadata and sim_dir
            # attributes, so we need to store them temporarily
            metadata = abd.metadata
            sim_dir = abd.sim_dir

            new_times = abd.t[abd.t > -100]
            abd = abd.interpolate(new_times)

            # Restore the metadata and sim_dir attributes
            abd.metadata = metadata
            abd.sim_dir = sim_dir

            # Undo the time shift
            abd.t += time_shift

        # Convert to the superrest frame
        abd_prime, best_BMS_transformation, best_rel_err = \
            abd.map_to_superrest_frame(t_0=t0)

        # Save to file
        with open(wf_path, 'wb') as f:
            pickle.dump(abd_prime, f)

    # Load from file
    with open(wf_path, 'rb') as f:
        abd_prime = pickle.load(f)

    return abd_prime


def rotate_wf(W, chi_f):
    """Rotates waveform to be aligned with the positive z-axis.

    Paremeters
    ----------
    W : WaveformModes object

    chi_f : array
        Remnant black hole spin vector in (x,y,z) directions.

    Returns
    -------
    W : WaveformModes object
        Rotated waveform.
    """
    th_z = np.arccos(chi_f[2]/np.linalg.norm(chi_f))
    r_dir = np.cross([0, 0, 1], chi_f)
    r_dir = th_z * r_dir / np.linalg.norm(r_dir)
    q = quaternion.from_rotation_vector(-r_dir)
    W.rotate_physical_system(q)
    return W


# Code below is for loading waveform data from existing files in a computer.

def load_EXTNR_data(ext_dir=None, wf_path=None, use_sxs=False,
                    sxs_id='SXS:BBH:0305', lev_N=6, ext_N=2):
    """Returns metadata, extrapolated waveform, and sxs id.

    Parameters
    __________
    ext_dir : string, optional [Default: None]
        Path to directory where simulation file is located.

    wf_path : string, optional [Default: None]
        File name of directory.

    use_sxs : bool [Default: False]
        If True, sxs.load will be used to load waveform from catalog instead of
        using a local file.

    sxs_id : string, optional [Default: 'SXS:BBH:0305']
        SXS ID of the simulation

    lev_N: int, optional [Default: 6]
        Numerical resolution label

    ext_N : int, optional [Default: 2]
        Extrapolation order label

    Example:
    load_NR_data("/Users/Username/Simulations/",
    "rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir")

    Returns
    _______
    md : Metadata

    W : WaveformModes
        Waveform object with peak at t=0M.

    sxs_id : string
        Name of this simulation
    """
    if use_sxs is False and ext_dir is None and wf_path is None:
        raise TypeError(
            'Set use_sxs=True or give directory and path name to the waveform '
            'data'
        )
    if use_sxs is False:
        md = sxs.metadata.Metadata.from_file(
            f'{ext_dir}metadata.txt', cache_json=False
        )
        W = scri.SpEC.read_from_h5(f'{ext_dir}{wf_path}')

        sxs_id = [name for name in md.alternative_names if 'SXS:BBH' in name]
        if (len(sxs_id) > 0):
            sxs_id = sxs_id[0]
        else:
            sxs_id = 'missing_sxs_id'

    trim_ind = W.max_norm_index()
    W.t = W.t - W.t[trim_ind]
    return md, W, sxs_id


def get_CCE_radii(simulation_dir, radius_index=None):
    """Returns CCE radii of a simulation.

    Parameters
    ----------
    simulation_dir : str
        Directory where simulation is located

    radius_index : int, optional [Default: None]

    Returns
    -------
    radii : list of strings
        Simulation radii
    """
    CCE_files = [
        filename for filename in os.listdir(simulation_dir)
        if 'rhOverM' in filename and '.h5' in filename
    ]
    radii = [filename.split('R')[1][:4] for filename in CCE_files]
    radii.sort(key=float)
    if radius_index is not None:
        return radii[radius_index:radius_index+1]
    else:
        return radii


def load_CCENR_data(cce_dir=None, file_format='SXS', use_sxs=False, N_sim=2):
    """Returns an AsymptoticBondiData object and CCE waveform.

    Example:
    load_CCENR_data("/Users/Username/Simulations/CCE_XXXX/LevX")

    Paremeters
    ----------
    cce_dir : str
        Directory where specific CCE waveform data is located

    file_format : str, optional [Default: 'SXS']
        'SXS' for data stored in .h5 format and 'RPXMB' for data store in both
        .h5 and .json formats.

    Returns
    -------
    abd_CCE : AsymptoticBondiData object.

    h_CCE : WaveformModes object
        Waveform object with peak at t=0M.
    """
    if use_sxs is False and cce_dir is None:
        raise TypeError(
            'Set use_sxs=True or give directory name to the waveform data'
        )
    if use_sxs is False:
        radius = get_CCE_radii(cce_dir)[0]
        abd_CCE = scri.SpEC.file_io.create_abd_from_h5(
            h=f'{cce_dir}rhOverM_BondiCce_R{radius}_CoM.h5',
            Psi4=f'{cce_dir}rMPsi4_BondiCce_R{radius}_CoM.h5',
            Psi3=f'{cce_dir}r2Psi3_BondiCce_R{radius}_CoM.h5',
            Psi2=f'{cce_dir}r3Psi2OverM_BondiCce_R{radius}_CoM.h5',
            Psi1=f'{cce_dir}r4Psi1OverM2_BondiCce_R{radius}_CoM.h5',
            Psi0=f'{cce_dir}r5Psi0OverM3_BondiCce_R{radius}_CoM.h5',
            file_format=file_format
        )
        h_CCE = abd_CCE.h
    else:
        abd_CCE = cce.load(N_sim)
        h_CCE = abd_CCE.h

    trim_ind = h_CCE.max_norm_index()
    h_CCE.t -= h_CCE.t[trim_ind]
    return abd_CCE, h_CCE
