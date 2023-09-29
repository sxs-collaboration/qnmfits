import json
import scri

from pathlib import Path
from urllib.request import urlretrieve

class cce:

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
        
        wf_paths = {}
        for keyword, argument in zip(['h', 'Psi4', 'Psi3', 'Psi2', 'Psi1', 'Psi0'], wf_types):
            wf_paths[keyword] = sim_dir / f'{wf}_BondiCce_R{R:04d}_CoM.h5'

        abd = scri.SpEC.create_abd_from_h5('RPXMB', **wf_paths)

        return abd, metadata
                
            




class cce_sim:

    def __init__(self, abd, metadata):

        self.abd = abd
        self.metadata = metadata