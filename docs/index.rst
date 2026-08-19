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
