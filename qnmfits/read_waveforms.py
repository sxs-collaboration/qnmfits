#!/usr/bin/env python

"""Read waveforms for modeling QNMss"""
import os

import spherical_functions as sf
import sxs
import scri
from scri.asymptotic_bondi_data import map_to_superrest_frame
import quaternion

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scri_qnm_modes import waveform_mismatch

def MT_to_WM(h_mts, dataType=scri.h):
    """Converts ModesTimeSeries object to a WaveformModes object.

    Parameters
    ----------
    h_mts : ModesTimeSeries object

    dataType : int, optional [Default: scri.h]
        `scri.dataType` appropriate for `data`

    Returns
    _______
    h : WaveformModes
        Waveform object
    
    """
    h = scri.WaveformModes(t=h_mts.t,\
                           data=np.array(h_mts)[:,sf.LM_index(abs(h_mts.s),-abs(h_mts.s),0):],\
                           ell_min=abs(h_mts.s),\
                           ell_max=h_mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
    h.r_is_scaled_out = True
    h.m_is_scaled_out = True
    return h

def WM_to_MT(h_wm):
    """Converts a WaveformModes object to a ModesTimesSeries object.

    Parameters
    ----------
    h_wm : WaveformModes
        Waveform object

    Returns
    _______
    h_mts : AsymptoticBondiData object
    
    """
    h_mts = scri.ModesTimeSeries(
        sf.SWSH_modes.Modes(
            h_wm.data,
            spin_weight=h_wm.spin_weight,
            ell_min=h_wm.ell_min,
            ell_max=h_wm.ell_max,
            multiplication_truncator=max,
        ),
        time=h_wm.t,
    )
    return h_mts

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
        Waveform object

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

def load_CCENR_data(cce_dir, file_format='SXS'):
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
    trim_ind = h_CCE.max_norm_index()
    h_CCE.t -= h_CCE.t[trim_ind]
    return abd_CCE, h_CCE

def to_superrest_frame(abd_CCE, t_0=350, padding_time=50, save=False,
                       sim_name=None):
    """Maps waveform into the BMS superrest frame of the remnant black hole.

    Parameters
    ----------
    abd_CCE : AsymptoticBondiData

    t_0 : float, optional [Default: 350.]
        Time to map to the superrest frame of the remnant black hole.

    padding_time : float, optional [Default: 50.]
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

    abd_superrest, transformations = map_to_superrest_frame(abd_CCE, t_0=t_0,
                                                            padding_time=padding_time)
    h_superrest = MT_to_WM(2.0*abd_superrest.sigma.bar)
    if save:
        scri.SpEC.file_io.write_to_h5(h_superrest, f"./BMS_data/h_{sim_name}_superrest.h5", 
                                      file_write_mode="w", attributes={}, use_NRAR_format=True)
    return h_superrest

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