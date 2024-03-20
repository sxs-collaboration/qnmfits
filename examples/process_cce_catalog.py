import numpy as np

import json
import shutil
import scri
import qnmfits

def abd_to_h(abd):
    """
    Convert an AsymptoticBondiData object to a WaveformModes object.
    
    Parameters
    ----------
    abd : AsymptoticBondiData
        The simulation data.
    
    Returns
    -------
    h_wm : WaveformModes
        The simulation data in the WaveformModes format.
    """
    # The strain is related to the shear in the following way
    h = 2*abd.sigma.bar

    # Convert to a WaveformModes object
    h_wm = scri.WaveformModes(
        t = h.t,
        data = np.array(h)[:,h.index(abs(h.s),-abs(h.s)):],
        ell_min = abs(h.s),
        ell_max = h.ell_max,
        frameType = scri.Inertial,
        dataType = scri.h,
        r_is_scaled_out = True,
        m_is_scaled_out = True,
    )

    return h_wm

def map_to_superrest(abd, t0):
    """
    Map an AsymptoticBondiData object to the superrest frame.

    Parameters
    ----------
    abd : AsymptoticBondiData
        The simulation data.

    t0 : float
        The time at which the superrest frame is defined. For ringdown
        studies, about 300M after the time of peak strain is recommended.

    Returns
    -------
    h_prime : WaveformModes
        The simulation data in the superrest frame.

    metadata : dict
        The superrest-frame remnant mass and spin.
    """
    # The extraction radius of the simulation
    R = int(abd.metadata['preferred_R'])

    # The directory of the data
    sim_dir = abd.sim_dir

    # Check if the transformation to the superrest frame has already been done
    wf_path = sim_dir / f'rhOverM_BondiCce_R{R:04d}_superrest.h5'
    if not wf_path.is_file():

        # Convert to a WaveformModes object to find time of peak strain
        h = abd_to_h(abd)
    
        # Shift the zero time to be at the peak of the strain
        time_shift = abd.t[np.argmax(h.norm())]
        abd.t -= time_shift

        # Window the data to speed up the transformation
        new_times = abd.t[abd.t > -100]
        abd = abd.interpolate(new_times)

        # Convert to the superrest frame
        abd_prime, _, _ = abd.map_to_superrest_frame(t_0=t0)

        # Undo the time shift
        abd.t += time_shift
        abd_prime.t += time_shift

        # Convert to a WaveformModes object
        h_prime = abd_to_h(abd_prime)

        # Save the WaveformModes object to file, and move to the data directory
        scri.SpEC.file_io.write_to_h5(h_prime, f'BondiCce_R{R:04d}_superrest.h5')
        shutil.move(f'rhOverM_BondiCce_R{R:04d}_superrest.h5', wf_path)

        # The AsymptoticBondiData object contains the remnant mass and spin.
        # Extract these and save to a metadata file.
        Mf = abd_prime.bondi_rest_mass()[-1]
        chif = abd_prime.bondi_dimensionless_spin()[-1]
        metadata = {'remnant_mass': Mf, 'remnant_dimensionless_spin': list(chif)}

        with open(sim_dir / f'metadata_BondiCce_R{R:04d}_superrest.json', 'w') as f:
            json.dump(metadata, f)

    # Load the WaveformModes
    h_prime = scri.SpEC.file_io.read_from_h5(wf_path.as_posix())

    # Load the metadata
    with open(sim_dir / f'metadata_BondiCce_R{R:04d}_superrest.json', 'r') as f:
        metadata = json.load(f)

    return h_prime, metadata

# -----------------------------------------------------------------------------

cce = qnmfits.cce()

# Transform each CCE waveform to the superrest frame and save to file
for ID in range(1, 14):
    abd = cce.load(ID)
    h_prime, metadata = map_to_superrest(abd, t0=300)