#!/usr/bin/env python

"""Read waveforms for modeling QNMss"""
import os

import spherical_functions as sf
import sxs
import scri
from scri.asymptotic_bondi_data.map_to_superrest_frame import MT_to_WM, map_to_superrest_frame
import quaternion

import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from qnmfits import mismatch

from .cce import cce
cce = cce()

def n_modes(ell_max, ell_min=2):
    """
    Calculates the number of spherical-harmonic modes between ell_min and
    ell_max.

    Parameters
    ----------
    ell_max : int
        The maximum value of ell.

    ell_min : int, optional
        The minimum value of ell. The default is 2.

    Returns
    -------
    int
        The number of spherical-harmonic modes between ell_min and ell_max.
    """
    return sum([2*ell+1 for ell in range(ell_min, ell_max+1)])

def to_WaveformModes(times, data, ell_max, ell_min=2):
    """
    Convert a dictionary of spherical-harmonic modes or a NumPy array to a
    WaveformModes object.

    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.

    data : dict or ndarray
        The spherical-harmonic waveform modes to convert to a WaveformModes
        object. If data is a dictionary, the keys are (ell,m) tuples and the
        values are the waveform data. If data is a NumPy array, the columns
        must correspond to the (ell,m) modes in the specific order required by
        scri: see the scri.WaveformModes documentation for details.

    ell_max : int
        The maximum value of ell included in the data.

    ell_min : int, optional
        The minimum value of ell included in the data. The default is 2.

    Returns
    -------
    h : WaveformModes
        The spherical-harmonic waveform modes in the WaveformModes format.
    """
    # Ensure data is in the correct format
    formatted_data = np.zeros((len(times), n_modes(ell_max, ell_min)), dtype=complex)

    if type(data) == dict:
        for i, (ell,m) in enumerate([(ell,m) for ell in range(ell_min, ell_max+1) for m in range(-ell,ell+1)]):
            if (ell,m) in data.keys():
                formatted_data[:,i] = data[ell,m]

    elif type(data) == np.ndarray:
        assert data.shape == (len(times), n_modes(ell_max, ell_min)), "Data array is not the correct shape."
        formatted_data = data

    # Convert to a WaveformModes object
    h = scri.WaveformModes(
        dataType = scri.h,
        t = times,
        data = formatted_data,
        ell_min = ell_min,
        ell_max = ell_max,
        frameType = scri.Inertial,
        r_is_scaled_out = True,
        m_is_scaled_out = True,
        )

    return h

def sxs_to_scri_WM(h_sxs, dataType=scri.h):
    """Converts an sxs WaveformModes object to that of scri.

    Parameters
    ----------
    h_sxs : sxs WaveformModes object

    dataType : int, optional [Default: scri.h]
        `scri.dataType` appropriate for `data`

    Returns
    _______
    h : WaveformModes
        Waveform object

    """
    h = scri.WaveformModes(t=h_sxs.t,\
                           data=h_sxs.data,\
                           ell_min=2,\
                           ell_max=h_sxs.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
    h.r_is_scaled_out = True
    h.m_is_scaled_out = True
    return h


def get_CCE_radii(simulation_dir, radius_index = None):
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
    CCE_files = [filename for filename in os.listdir(simulation_dir) if 'rhOverM' in filename and '.h5' in filename]
    radii = [filename.split('R')[1][:4] for filename in CCE_files]
    radii.sort(key=float)
    if radius_index != None:
        return radii[radius_index:radius_index+1]
    else:
        return radii

def load_EXTNR_data(ext_dir=None, wf_path=None, use_sxs=False,
        sxs_id='SXS:BBH:0305', lev_N=6, ext_N=2):
    """Returns metadata structure, WaveformModes object, and sxs_id

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
    if use_sxs == False and ext_dir == None and wf_path == None:
        raise TypeError('Set use_sxs=True or give directory and path name to'\
                ' the waveform data')
    if use_sxs == False:
        md = sxs.metadata.Metadata.from_file(f'{ext_dir}metadata.txt', cache_json=False)
        W = scri.SpEC.read_from_h5(f'{ext_dir}{wf_path}')

        sxs_id = [name for name in md.alternative_names if 'SXS:BBH' in name]
        if (len(sxs_id) > 0):
            sxs_id = sxs_id[0]
        else:
            sxs_id = 'missing_sxs_id'
    else:
        catalog = sxs.load("catalog")
        md = sxs.load(f"{sxs_id}/Lev{lev_N}/metadata.json")
        W = sxs.load(f"{sxs_id}/Lev{lev_N}/rhOverM", extrapolation_order=ext_N)

    trim_ind = W.max_norm_index()
    W.t = W.t - W.t[trim_ind]
    return md, W, sxs_id

def load_CCENR_data(cce_dir=None, file_format='SXS', use_sxs=False, N_sim=2):
    """Returns an AsymptoticBondiData object and WaveformModes object. 

    Paremeters
    ----------
    cce_dir : str
        Directory where specific CCE waveform data is located

    file_format: str, optional [Default: 'SXS']
        'SXS' for data stored in .h5 format and 'RPXMB' for data store in both
        .h5 and .json formats.

    Example:
    load_CCENR_data("/Users/Username/Simulations/CCE_XXXX/LevX")
    
    Returns
    -------
    abd_CCE : AsymptoticBondiData

    h_CCE : WaveformModes
        Waveform object with peak at t=0M

    """
    if use_sxs == False and cce_dir == None:
        raise TypeError('Set use_sxs=True or give directory name to'\
                ' the waveform data')
    if use_sxs == False: 
        radius = get_CCE_radii(cce_dir)[0]
        abd_CCE = scri.SpEC.file_io.create_abd_from_h5(\
                        h = f'{cce_dir}rhOverM_BondiCce_R{radius}_CoM.h5',
                        Psi4 = f'{cce_dir}rMPsi4_BondiCce_R{radius}_CoM.h5',
                        Psi3 = f'{cce_dir}r2Psi3_BondiCce_R{radius}_CoM.h5',
                        Psi2 = f'{cce_dir}r3Psi2OverM_BondiCce_R{radius}_CoM.h5',
                        Psi1 = f'{cce_dir}r4Psi1OverM2_BondiCce_R{radius}_CoM.h5',
                        Psi0 = f'{cce_dir}r5Psi0OverM3_BondiCce_R{radius}_CoM.h5',
                        file_format = file_format)
        h_CCE = MT_to_WM(2.0*abd_CCE.sigma.bar)
    else:
        abd_CCE = cce.load(N_sim)
        h_CCE = MT_to_WM(2*abd_CCE.sigma.bar)

    trim_ind = h_CCE.max_norm_index()
    h_CCE.t -= h_CCE.t[trim_ind]
    return abd_CCE, h_CCE

def to_superrest_frame(abd_CCE, t_0=350., padding_time=100, save=False,
                       sim_name=None):
    """Maps waveform into the BMS superrest frame of the remnant black hole.

    Parameters
    ----------
    abd_CCE : AsymptoticBondiData

    t_0 : float, optional [Default: 350.]
        Time to map to the superrest frame of the remnant black hole. For ringdown
        studies, about 300M after the time of peak strain is recommended.

    padding_time : float, optional [Default: 100.]
        Amount by which to pad around t_0 to speed up computations.

   save : bool, optional [Default: False]
        If True, the supertranslated waveform will be saved in a directory
        called 'BMS_data'.

    sim_name : str, optional [Default: None]
        Use if saving the waveform. Format is h_{sim_name}_superrest.h5

    Returns
    -------
    h_superrest : WaveformModes
        WaveformModes object in the BMS superrest frame

    """

    # The extraction radius of the simulation
    R = int(abd_CCE.metadata['preferred_R'])

    # Check if the transformation to the superrest frame has already been done
    wf_path = abd_CCE.sim_dir / f'rhoverM_BondiCce_R{R:04d}_t0{t_0}_superrest.pickle'

    if not wf_path.is_file():
        abd_superrest, transformations = map_to_superrest_frame(abd_CCE, t_0=t_0,
                                                        padding_time=padding_time)
        # Save to file
        with open(wf_path, 'wb') as f:
            pickle.dump(abd_superrest, f)

    # Load from file
    else: 
        with open(wf_path, 'rb') as f:
            abd_superrest = pickle.load(f)
   
    h_superrest = MT_to_WM(2.0*abd_superrest.sigma.bar)
    return abd_superrest, h_superrest 

def get_resolution_mismatches(W, W_LR, t0_arr, mode=None, news=False): 
    '''Waveforms are assumed to have z-axis aligned by final spin or something else, 
    and times aligned as well. Rotation about z-axis is done to minimize mismatch. 

    Parameters
    ----------
    W : WaveformModes
        Waveform object

    W_LR : WaveformModes
        Waveform object of lower resolution

    t0_arr : float array
        Waveform model is 0 for t < t0_arr values

    mode : tuple, optional [Default: None]
        (ell, m) mode tuple to calculate specific mode mismatch instead of all-mode
        mismatch.

    news : bool, optional [Default: False]
        Use the strain or news domain to calculate mismatches
        
    Returns
    -------
    resolution_mismatch : list of floats
        Minimized NR resolution mismatch at each time, t0
    
    rotation_LR : list of floats
        Minimized phi rotations at each time, t0
        
    '''
    resolution_mismatch = []
    rotation_LR = []
    W_interp = W.copy() 
    W_interp.data = CubicSpline(W_LR.t, W_LR.data.real)(W.t) \
                    + 1.j*CubicSpline(W_LR.t, W_LR.data.imag)(W.t) 
    for t0 in t0_arr: 
        W_shifted = W.copy()[np.argmin(abs(W.t - t0)):np.argmin(abs(W.t - 90))+1,:] 
        W_shifted.t = W_shifted.t - W_shifted.t[0] 
        W_LR_shifted = W_interp.copy()[np.argmin(abs(W_interp.t - t0))
                                       :np.argmin(abs(W_interp.t - 90))+1,:] 
        W_LR_shifted.t = W_LR_shifted.t - W_LR_shifted.t[0]
        if news == True:
            W_shifted.data = W_shifted.data_dot
            W_shifted.dataType = scri.hdot
            W_LR_shifted.data = W_LR_shifted.data_dot
            W_LR_shifted.dataType = scri.hdot
        def mism(phi):
            q = quaternion.from_rotation_vector([0,0,phi])
            W_rot = W_LR_shifted.copy()
            W_rot.rotate_physical_system(q);
            return waveform_mismatch(W_shifted, W_rot, 0., mode)
        
        res = minimize_scalar(mism, bounds=(0,2*np.pi))
        resolution_mismatch.append(res.fun)
        rotation_LR.append(res.x)
        
    return resolution_mismatch, rotation_LR

def align_lev(W, W_LR, t0, mode=None, news=False):
    """Rotate low resolution waveform to match the higher resolution waveform.

    Parameters
    ----------
    W : WaveformModes
        Waveform object

    W_LR : WaveformModes
        Waveform object of lower resolution

    t0 : float
        Waveform model is 0 for t < t0
    
    mode : tuple, optional [Default: None]
        (ell, m) mode tuple to calculate specific mode mismatch instead of all-mode
        mismatch.

    news : bool, optional [Default: False]
        Use the strain or news domain to calculate mismatches
    
    Returns
    -------
    W_rot : WaveformModes
        Rotated waveform object

    """
    mism, phi = get_resolution_mismatches(W, W_LR, [t0], mode, news)
    W_interp = W.copy() 
    W_interp.data = CubicSpline(W_LR.t, W_LR.data.real)(W.t) \
                    + 1.j*CubicSpline(W_LR.t, W_LR.data.imag)(W.t) 
    q = quaternion.from_rotation_vector([0,0,phi[0]])
    W_rot = W_interp.copy()
    W_rot.rotate_physical_system(q);
    return W_rot

def rotate_wf(W, chi_f):
    """Rotates waveform to be aligned with the positive z-axis.

    Paremeters
    ----------
    W : WaveformModes
        Waveform object

    chi_f : array
        Remnant black hole spin in x, y, and z directions

    Returns
    -------
    W : WaveformModes
        Rotated waveform object

    """
    th_z = np.arccos(chi_f[2]/np.linalg.norm(chi_f))
    r_dir = np.cross([0,0,1],chi_f)
    r_dir = th_z * r_dir / np.linalg.norm(r_dir)
    q = quaternion.from_rotation_vector(-r_dir)
    W.rotate_physical_system(q);
    return W
