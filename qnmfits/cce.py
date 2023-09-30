import numpy as np

import json
import scri
import pickle

from pathlib import Path
from urllib.request import urlretrieve

class cce:
    """
    A class for loading simulations from the CCE catalog.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        # The directory of this file
        self.file_dir = Path(__file__).resolve().parent
        
        # Load the CCE catalog JSON
        with open(self.file_dir / 'data/cce-catalog.json', 'r') as f:
            self.catalog = json.load(f)
        
    def load(self, ID):
        """
        Load a simulation from the catalog, and add the simulation directory
        and metadata to the AsymptoticBondiData object.

        Parameters
        ----------
        ID : int
            The ID of the simulation to load.

        Returns
        -------
        abd : AsymptoticBondiData
            The simulation data.
        """
        # Convert the ID to the simulation name
        name = f'SXS:BBH_ExtCCE:{int(ID):04d}'

        # The location we'll download the files to
        sim_dir = self.file_dir / 'data' / name
        sim_dir.mkdir(exist_ok=True)

        # Access the appropriate entry in the CCE catalog
        for entry in self.catalog:
            if entry['name'] == name:
                metadata = entry
        
        # Extract useful information
        url = metadata['url']
        R = int(metadata['preferred_R'])

        metadata_path = sim_dir / 'metadata.json'

        # Check if simulation metadata already exists
        if not metadata_path.is_file():
            # Download simulation metadata
            print(f'Downloading {url}/files/Lev5/metadata.json')
            urlretrieve(
                f'{url}/files/Lev5/metadata.json?download=1', 
                metadata_path
                )
        
        # Load and merge with existing metadata
        with open(sim_dir / 'metadata.json', 'r') as f:
            official_metadata = json.load(f)
        metadata = {**metadata, **official_metadata}

        # Download simulation data

        # Strain and Weyl scalar names
        wf_types = ['rhOverM', 'rMPsi4', 'r2Psi3', 'r3Psi2OverM', 'r4Psi1OverM2', 'r5Psi0OverM3']

        for wf in wf_types:

            wf_path = sim_dir / f'{wf}_BondiCce_R{R:04d}_CoM.h5'

            # Check if the simulation data already exists
            if not wf_path.is_file():
                # Download simulation data                
                print(f'Downloading {url}/files/Lev5/{wf}_BondiCce_R{R:04d}_CoM.h5')
                urlretrieve(
                    f'{url}/files/Lev5/{wf}_BondiCce_R{R:04d}_CoM.h5?download=1',
                    wf_path
                    ) 
            
            wf_json_path = sim_dir / f'{wf}_BondiCce_R{R:04d}_CoM.json'
            # Check if the simulation json data already exists
            if not wf_json_path.is_file():
                print(f'Downloading {url}/files/Lev5/{wf}_BondiCce_R{R:04d}_CoM.json')
                urlretrieve(
                    f'{url}/files/Lev5/{wf}_BondiCce_R{R:04d}_CoM.json?download=1',
                    wf_json_path
                    ) 
        
        # Get a dictionary of paths for each file, and create the 
        # AsymptoticBondiData object
        wf_paths = {}
        for keyword, argument in zip(['h', 'Psi4', 'Psi3', 'Psi2', 'Psi1', 'Psi0'], wf_types):
            wf_paths[keyword] = sim_dir / f'{argument}_BondiCce_R{R:04d}_CoM.h5'

        abd = scri.SpEC.create_abd_from_h5(file_format='RPXMB', **wf_paths)
        
        # Store the simulation directory and metadata in the object
        abd.sim_dir = sim_dir
        abd.metadata = metadata

        return abd
    

    def abd_to_h(self, abd):
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
            dataType = scri.h,
            t = h.t,
            data = np.array(h)[:,h.index(abs(h.s),-abs(h.s)):],
            ell_min = 2,
            ell_max = h.ell_max,
            frameType = scri.Inertial,
            r_is_scaled_out = True,
            m_is_scaled_out = True,
        )

        return h_wm


    def map_to_superrest(self, abd):
        """
        Map an AsymptoticBondiData object to the superrest frame.

        Parameters
        ----------
        abd : AsymptoticBondiData
            The simulation data.

        Returns
        -------
        abd_prime : AsymptoticBondiData
            The simulation data in the superrest frame.
        """
        # The extraction radius of the simulation
        R = abd.metadata['preferred_R']

        # Check if the transformation to the superrest frame has already been
        # done
        wf_path = abd.sim_dir / f'rhoverM_BondiCce_R{R:04d}_superrest.pickle'

        if not wf_path.is_file():

            # Convert to a WaveformModes object to find time of peak strain
            h = self.abd_to_h(abd)
        
            # Shift the zero time to be at the peak of the strain
            abd.t -= abd.t[np.argmax(h.norm())]

            # Convert to the superrest frame
            abd_prime, transformations = abd.map_to_superrest_frame(t_0=300)

            # Save to file
            with open(wf_path, 'wb') as f:
                pickle.dump(abd_prime, f)

        # Load from file
        with open(wf_path, 'rb') as f:
            abd_prime = pickle.load(f)

        return abd_prime