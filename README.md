# ndfilters

[![tests](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/ndfilters/graph/badge.svg?token=BFTOVSyFtf)](https://codecov.io/gh/sun-data/ndfilters)
[![Black](https://github.com/sun-data/ndfilters/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/ndfilters/actions/workflows/black.yml)
[![Documentation Status](https://readthedocs.org/projects/ndfilters/badge/?version=latest)](https://ndfilters.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ndfilters.svg)](https://badge.fury.io/py/ndfilters)

Similar to the filters in `scipy.ndimage` but accelerated using `numba`.

## Installation

`ndfilters` is published on PyPI and can be installed using `pip`.

```bash
pip install ndfilters
```

## Examples

The only filter currently implemented is a [trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_autosummary/ndfilters.trimmed_mean_filter.html#ndfilters.trimmed_mean_filter).
This filter ignores a given portion of the dataset before calculating the mean at each pixel.

![trimmed mean filter](https://ndfilters.readthedocs.io/en/latest/_images/ndfilters.trimmed_mean_filter_0_1.png)
