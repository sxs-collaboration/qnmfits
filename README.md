# Welcome to qnmfits

`qnmfits` is an open-source Python code for least-squares fitting of quasinormal modes to ringdown waveforms.
The code is designed to interface with the [`sxs`](https://github.com/sxs-collaboration/sxs) and [`scri`](https://github.com/moble/scri) packages, which allows for easy loading and transformations of SXS waveforms (including CCE waveforms) - but any waveform data can be used.
Quasinormal-mode frequencies and spherical-spheroidal mixing coefficients are loaded via the [`qnm`](https://github.com/duetosymmetry/qnm) package, and multimode fitting across multiple spherical-harmonic modes with mode mixing is handled automatically.
<!-- Additionally, it is equipped with a 'greedy' algorithm that picks the most important modes to model based on their power contribution to the residual between numerical and model waveforms.  -->

## Installation

qnmfits is available on [conda-forge](https://anaconda.org/conda-forge/qnmfits):

```bash
conda install conda-forge::qnmfits
```

 and [PyPI](https://pypi.org/project/qnmfits/):

```bash
pip install qnmfits
```

### From source

Clone and install this code locally:

```bash
git clone git@github.com:sxs-collaboration/qnmfits.git
cd qnmfits
pip install .
```

<!-- ## Documentation

Automatically-generated API documentation is available on [Read the Docs: qnmfits](https://qnmfits.readthedocs.io/). -->

## Usage

Perform a seven-overtone fit to the (2,2) mode of SXS:BBH_ExtCCE:0001:

```python
import numpy as np
import qnmfits

# Download SXS:BBH_ExtCCE:0001
abd = qnmfits.cce.load(1)

# Transform to the superrest frame. First, shift the times for convenience:
h22 = abd.h.data[:, abd.h.index(2, 2)]
abd.t -= abd.t[np.argmax(np.abs(h22))]

# Map to the superrest frame
abd_prime = qnmfits.utils.to_superrest_frame(abd, t0=300)

# Get the strain and remnant mass and spin
h = abd_prime.h
Mf = abd_prime.bondi_rest_mass()[-1]
chif = np.linalg.norm(abd_prime.bondi_dimensionless_spin()[-1])

# QNMs we want to fit for. The format is (ell, m, n, sign), where sign is +1
# for "regular" (positive real part) modes, and -1 is for "mirror" (negative
# real part) modes.
qnms = [(2, 2, n, 1) for n in range(7+1)]

# Spherical modes we want to fit to. The format is (ell, m).
spherical_modes = [(2, 2)]

# Ringdown start time
t0 = 0.

# Perform the fit
best_fit = qnmfits.fit(
    data=h,
    chif=chif,
    Mf=Mf,
    qnms=qnms,
    spherical_modes=spherical_modes,
    t0=t0
)

print(f"Mismatch = {best_fit['mismatch']}")
```

The mismatch should be of order 1E-7.
Please see the notebooks in `examples` for additional code usage.

## Contributing

Contributions are welcome! There are at least two ways to contribute to this codebase:

1. If you find a bug or want to suggest an enhancement, use the [issue tracker](https://github.com/sxs-collaboration/qnmfits/issues) on GitHub. It's a good idea to look through past issues, too, to see if anybody has run into the same problem or made the same suggestion before.
2. If you will write or edit the python code, we use the [fork and pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) model.

You are also allowed to make use of this code for other purposes, as detailed in the [MIT license](LICENSE). For any type of contribution, please follow the [code of conduct](CODE_OF_CONDUCT.md).

<!-- ## Citing this code

If this package contributes to a project that leads to a publication,
please acknowledge this by citing the `qnmfits` article in. -->

## Credits

The code is developed and maintained by [Lorena Magaña Zertuche](https://github.com/lmagana3) and [Eliot Finch](https://github.com/eliotfinch).
