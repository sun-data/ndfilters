# ndfilters

[![tests](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/ndfilters/graph/badge.svg?token=BFTOVSyFtf)](https://codecov.io/gh/sun-data/ndfilters)
[![Black](https://github.com/sun-data/ndfilters/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/ndfilters/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/ndfilters/badge/?version=latest)](https://ndfilters.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ndfilters.svg)](https://badge.fury.io/py/ndfilters)

Similar to the filters in `scipy.ndimage` but accelerated and parallelized using 
[Numba](https://numba.readthedocs.io/en/stable/).

## Installation

`ndfilters` is published on PyPI and can be installed using `pip`.

```bash
pip install ndfilters
```

## Gallery

### Mean filter

The [mean filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.mean_filter.html#ndfilters.mean_filter)
calculates a multidimensional rolling mean for the given kernel shape.

![mean filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.mean_filter_0_0.png)

### Trimmed mean filter

The  [trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.trimmed_mean_filter.html#ndfilters.trimmed_mean_filter)
is like the mean filter except it ignores a given portion of the dataset before calculating the mean at each pixel.

![trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.trimmed_mean_filter_0_0.png)

### Variance filter

The [variance filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.variance_filter.html#ndfilters.variance_filter)
calculates the rolling variance for the given kernel shape.

![variance filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.variance_filter_0_0.png)
