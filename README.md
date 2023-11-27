# Welcome to qnmfits

`qnmfits` is an open-source Python code for multimode ringdown modeling. It 
allows you to fit for the quasinormal mode amplitudes by minimizing the 
mismatch between a numerical relativity waveform and an analytical ringdown 
model. Additionally, it is equipped with a 'greedy' algorithm that picks the 
most important modes to model based on their power contribution to the residual 
between numerical and model waveforms. This multimode modeling code includes
overtones, retrograde modes, and spherical-spheroidal mixing coefficients. 
`qnmfits` uses the parent [`qnm`](https://github.com/duetosymmetry/qnm) package to 
access quasinormal mode frequencies and spherical-spheroidal mixing
coefficients.

This code can handle both extrapolated and CCE waveforms from the SXS catalog,
and it comes equipped with [`scri`](https://github.com/moble/scri/) based
function that allows you to map the CCE waveform to the superrest frame of the
remnant for a high-precision ringdown analysis.

## Installation
This packages uses [`scri`](https://github.com/moble/scri), which should be installed with `conda` (see the `scri` quickstart [here](https://github.com/moble/scri#quick-start)). It is recommended to install `scri` first, before installing this package:

```bash
conda install -c conda-forge scri
```

### PyPI
_**qnmfits**_ is available on [PyPI](https://pypi.org/project/qnmfits/):

```shell
pip install qnmfits
```

### Conda
_**qnm**_ is available on [conda-forge](https://anaconda.org/conda-forge/qnm):

```shell
conda install -c conda-forge qnm
```

### From source
Clone and install this code locally:

```bash
git clone git@github.com:sxs-collaboration/qnmfits.git
cd qnmfits
pip install .
```

## Dependencies
This package uses the following dependencies:

* [`qnm`](https://github.com/duetosymmetry/qnm)
* [`sxs`](https://github.com/sxs-collaboration/sxs)
* [`scri`](https://github.com/moble/scri)

Dependencies should be installed automatically by using the instructions above.

## Documentation
Automatically-generated API documentation is available on [Read the Docs: qnmfits](https://qnmfits.readthedocs.io/).

## Usage


## Contributing
Contributions are welcome! There are at least two ways to contribute to this codebase:

1. If you find a bug or want to suggest an enhancement, use the [issue tracker](https://github.com/sxs-collaboration/qnmfits/issues) on GitHub. It's a good idea to look through past issues, too, to see if anybody has run into the same problem or made the same suggestion before.
2. If you will write or edit the python code, we use the [fork and pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) model.

You are also allowed to make use of this code for other purposes, as detailed in the [MIT license](LICENSE). For any type of contribution, please follow the [code of conduct](CODE_OF_CONDUCT.md).


## Citing this code
If this package contributes to a project that leads to a publication,
please acknowledge this by citing the `qnmfits` article in.

## Credits
The code is developed and maintained by [Lorena Magaña Zertuche](https://github.com/lmagana3) and [Eliot Finch](https://github.com/eliotfinch).
