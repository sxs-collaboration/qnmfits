import numpy as np

import json
import scri

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
        
    def load(self, ID, level=5):
        """
        Load a simulation from the catalog, and add the simulation directory
        and metadata to the AsymptoticBondiData object.

        Parameters
        ----------
        ID : int
            The ID of the simulation to load.

        level : int, optional
            The level of the simulation to load. Default is 5.

        Returns
        -------
        abd : AsymptoticBondiData
            The simulation data.
        """
        # Convert the ID to the simulation name
        name = f'SXS:BBH_ExtCCE:{int(ID):04d}'

        # The location we'll download the files to
        sim_dir = self.file_dir / 'data' / name / f'Lev{level}'
        sim_dir.mkdir(exist_ok=True, parents=True)

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
            print(f'Downloading {url}/files/Lev{level}/metadata.json')
            urlretrieve(
                f'{url}/files/Lev{level}/metadata.json?download=1', 
                metadata_path
                )
        
        # Load and merge with existing metadata
        with open(metadata_path, 'r') as f:
            official_metadata = json.load(f)
        metadata = {**metadata, **official_metadata, **{'level': level}}

        # Download simulation data

        # Strain and Weyl scalar names
        wf_types = ['rhOverM', 'rMPsi4', 'r2Psi3', 'r3Psi2OverM', 'r4Psi1OverM2', 'r5Psi0OverM3']

        for wf in wf_types:

            wf_path = sim_dir / f'{wf}_BondiCce_R{R:04d}.h5'

            # Check if the simulation data already exists
            if not wf_path.is_file():
                # Download simulation data                
                print(f'Downloading {url}/files/Lev{level}:{wf}_BondiCce_R{R:04d}.h5')
                urlretrieve(
                    f'{url}/files/Lev{level}:{wf}_BondiCce_R{R:04d}.h5?download=1',
                    wf_path
                    ) 
            
            wf_json_path = sim_dir / f'{wf}_BondiCce_R{R:04d}.json'
            # Check if the simulation json data already exists
            if not wf_json_path.is_file():
                print(f'Downloading {url}/files/Lev{level}:{wf}_BondiCce_R{R:04d}.json')
                urlretrieve(
                    f'{url}/files/Lev{level}:{wf}_BondiCce_R{R:04d}.json?download=1',
                    wf_json_path
                    ) 
        
        # Get a dictionary of paths for each file, and create the 
        # AsymptoticBondiData object
        wf_paths = {}
        for keyword, argument in zip(['h', 'Psi4', 'Psi3', 'Psi2', 'Psi1', 'Psi0'], wf_types):
            wf_paths[keyword] = sim_dir / f'{argument}_BondiCce_R{R:04d}.h5'

        abd = scri.SpEC.create_abd_from_h5(file_format='RPDMB', **wf_paths)
        
        # Store the simulation directory and metadata in the object
        abd.sim_dir = sim_dir
        abd.metadata = metadata

        return abd
