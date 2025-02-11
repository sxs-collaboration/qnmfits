![PyPI - Version](https://img.shields.io/pypi/v/qnmfits)
![Conda Version](https://img.shields.io/conda/v/conda-forge/qnmfits)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14806974.svg)](https://doi.org/10.5281/zenodo.14806974)

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

## A note on QNM labeling

In this package QNMs are specified with four numbers: `(ell, m, n, sign)`. The first three numbers refer to the usual angular (`ell`), azimuthal (`m`), and overtone (`n`) indices. The fourth number is either `+1` or `-1`, and refers to the sign of the real part of the QNM frequency. In other words, `sign=1` refers to the "regular" QNMs to the right of the imaginary axis, and `sign=-1` refers to "mirror" QNMs to the left of the imaginary axis. Note that this is different to the prograde (co-rotating) and retrograde (counter-rotating) classification you sometimes see.

For data which the `qnm` package can't compute (for example, the special $(2,2,8)$ "multiplet"), additional data can be downloaded with the code
```python
import qnmfits
qnmfits.download_cook_data()
```
which downloads the Cook & Zalutskiy data from [here](https://zenodo.org/records/10093311).
Note that there are different labelling conventions for these multiplets. For example, the Schwarzschild $(2,2,8)$ QNM has the behaviour of "splitting" into two branches when the spin is increased:

![QNM multiplet taxonomy](https://github.com/eliotfinch/qnmfits/raw/main/examples/qnm_multiplet_taxonomy.png)

This has led to these two branches being labelled as $(2,2,8_0)$ and $(2,2,8_1)$ by Cook & Zalutskiy ([arxiv:1607.07406](http://arxiv.org/abs/1607.07406)). However, from a practical perspective we will be mostly working with Kerr black holes, and these two branches behave as a $n=8$ and $n=9$ overtone. So, as indicated by the figure above, we label them as such (this follows the convention of Forteza & Mourier ([arXiv:2107.11829](http://arxiv.org/abs/2107.11829))).

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
