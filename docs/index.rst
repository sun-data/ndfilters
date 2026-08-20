Introduction
============

:mod:`ndfilters` is a library of n-dimensional image filters similar to those
in :mod:`scipy.ndimage`, but accelerated and parallelized using the
`Numba <https://numba.readthedocs.io/en/stable/>`_ just-in-time compiler.

Compared to their :mod:`scipy.ndimage` equivalents, the filters in this
library offer some additional capabilities:

- **Axis selection.** Every filter accepts an ``axis`` argument, so the
  kernel can be applied to any subset of the array's axes while the remaining
  axes act as batch dimensions.
- **Masking.** A boolean ``where`` mask excludes selected elements of the
  input array from the calculation.
- **Physical units.** Inputs can be either :class:`numpy.ndarray` or
  :class:`astropy.units.Quantity` instances.
- **Varying kernels.** The convolution kernel is allowed to change along axes
  orthogonal to the convolution axes.

The following filters are currently implemented:

- :func:`ndfilters.mean_filter`, a multidimensional rolling mean.
- :func:`ndfilters.trimmed_mean_filter`, a rolling mean that ignores a given
  portion of the dataset at each pixel.
- :func:`ndfilters.median_filter`, a multidimensional rolling median.
- :func:`ndfilters.variance_filter`, a multidimensional rolling variance.
- :func:`ndfilters.generic_filter`, a rolling filter that applies an
  arbitrary compiled function to each kernel footprint.
- :func:`ndfilters.convolve`, a multidimensional convolution with support for
  spatially-varying kernels.


Differences from :mod:`scipy.ndimage`
=====================================

Where a filter in this library has a :mod:`scipy.ndimage` counterpart, the
two agree except in the following cases.

- **Boundary modes.** Only ``"mirror"``, ``"nearest"``, and ``"wrap"`` are
  supported, plus ``"truncate"``, which has no :mod:`scipy.ndimage`
  equivalent and simply drops the parts of the kernel that fall outside the
  array. SciPy's ``"reflect"``, ``"constant"``, and ``"grid-*"`` modes raise
  a :class:`ValueError` here.
- **Integer input.** Integer arrays are promoted to floating point, so the
  result is a float and is not truncated. :mod:`scipy.ndimage` returns the
  dtype of the input, and for the separable filters it truncates its
  intermediates as well. The promotion is what lets a ``where`` mask that
  excludes an entire kernel footprint return :obj:`numpy.nan`.
- **Even-sized median footprints.** :func:`ndfilters.median_filter` averages
  the two middle elements, like :func:`numpy.median`, while
  :func:`scipy.ndimage.median_filter` selects the element of rank
  ``size // 2``, the larger of the two. SciPy's convention keeps the result
  in the dtype of the input and never introduces a value that was not
  already in the footprint, but it is a biased estimator: on unit-variance
  noise a ``size=2`` filter shifts the signal by roughly ``0.57``. The two
  conventions agree exactly for odd-sized footprints.


Installation
============
:mod:`ndfilters` is published on PyPI and can be installed using::

    pip install ndfilters


Quickstart
==========

Every filter takes an array and the shape of the kernel, and returns the
filtered array.
As an example, here is a median filter applied to a sample image:

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import scipy.datasets
    import ndfilters

    img = scipy.datasets.ascent()
    img_filtered = ndfilters.median_filter(img, size=21)

    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    axs[0].set_title("original image");
    axs[0].imshow(img, cmap="gray");
    axs[1].set_title("filtered image");
    axs[1].imshow(img_filtered, cmap="gray");

See the documentation of each filter in the API reference below for more
examples.


API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    ndfilters



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
