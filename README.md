# ndfilters

[![tests](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/ndfilters/graph/badge.svg?token=BFTOVSyFtf)](https://codecov.io/gh/sun-data/ndfilters)
[![Black](https://github.com/sun-data/ndfilters/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/ndfilters/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/ndfilters/badge/?version=latest)](https://ndfilters.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ndfilters.svg)](https://badge.fury.io/py/ndfilters)

`ndfilters` is a library of n-dimensional image filters similar to those in
[`scipy.ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html),
but accelerated and parallelized using
[Numba](https://numba.readthedocs.io/en/stable/).

Compared to their `scipy.ndimage` equivalents, the filters in this library
offer some additional capabilities:

- **Axis selection.** Every filter accepts an `axis` argument, so the kernel
  can be applied to any subset of the array's axes while the remaining axes
  act as batch dimensions.
- **Masking.** A boolean `where` mask excludes selected elements of the input
  array from the calculation.
- **Physical units.** Inputs can be either `numpy.ndarray` or
  `astropy.units.Quantity` instances.
- **Varying kernels.** The convolution kernel is allowed to change along axes
  orthogonal to the convolution axes.

## Differences from `scipy.ndimage`

Where a filter in this library has a `scipy.ndimage` counterpart, the two
agree except in the following cases.

- **Boundary modes.** Only `"mirror"`, `"nearest"`, and `"wrap"` are
  supported, plus `"truncate"`, which has no `scipy.ndimage` equivalent and
  simply drops the parts of the kernel that fall outside the array. SciPy's
  `"reflect"`, `"constant"`, and `"grid-*"` modes raise a `ValueError` here.
- **Integer input.** Integer arrays are promoted to floating point, so the
  result is a float and is not truncated. `scipy.ndimage` returns the dtype
  of the input, and for the separable filters it truncates its intermediates
  as well. The promotion is what lets a `where` mask that excludes an entire
  kernel footprint return `NaN`.
- **Even-sized median footprints.** `ndfilters.median_filter` averages the
  two middle elements, like `numpy.median`, while
  `scipy.ndimage.median_filter` selects the element of rank `size // 2`, the
  larger of the two. SciPy's convention keeps the result in the dtype of the
  input and never introduces a value that was not already in the footprint,
  but it is a biased estimator: on unit-variance noise a `size=2` filter
  shifts the signal by roughly `0.57`. The two conventions agree exactly for
  odd-sized footprints.

The full documentation is hosted on [Read the Docs](https://ndfilters.readthedocs.io/en/latest/).

## Installation

`ndfilters` is published on PyPI and can be installed using `pip`.

```bash
pip install ndfilters
```

## Quickstart

Every filter takes an array and the shape of the kernel, and returns the
filtered array.

```python
import scipy.datasets
import ndfilters

img = scipy.datasets.ascent()
img_filtered = ndfilters.median_filter(img, size=21)
```

## Gallery

### Mean filter

The [mean filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.mean_filter.html#ndfilters.mean_filter)
calculates a multidimensional rolling mean for the given kernel shape.

![mean filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.mean_filter_0_0.png)

### Trimmed mean filter

The [trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.trimmed_mean_filter.html#ndfilters.trimmed_mean_filter)
is like the mean filter except it ignores a given portion of the dataset before calculating the mean at each pixel.

![trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.trimmed_mean_filter_0_0.png)

### Median filter

The [median filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.median_filter.html#ndfilters.median_filter)
calculates a multidimensional rolling median for the given kernel shape.

![median filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.median_filter_0_0.png)

### Variance filter

The [variance filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.variance_filter.html#ndfilters.variance_filter)
calculates the rolling variance for the given kernel shape.

![variance filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.variance_filter_0_0.png)

### Generic filter

The [generic filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.generic_filter.html#ndfilters.generic_filter)
applies an arbitrary compiled function to each kernel footprint.
It is the engine behind the other rolling filters in this library, and it can
be used directly to build custom filters.

![generic filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.generic_filter_0_0.png)

### Convolution

[`ndfilters.convolve()`](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.convolve.html#ndfilters.convolve)
convolves an array with a given kernel.
Unlike `scipy.ndimage.convolve()` or `astropy.convolution.convolve()`,
the kernel is allowed to vary along axes orthogonal to the convolution axes.

![convolve](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.convolve_0_2.png)
