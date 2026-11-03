from typing import Literal
import numpy as np
import numba
import astropy.units as u
import ndfilters
from ._select import select

__all__ = [
    "median_filter",
]


def median_filter(
    array: np.ndarray | u.Quantity,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
) -> np.ndarray:
    """
    Calculate a multidimensional rolling median.

    Parameters
    ----------
    array
        The input array to be filtered
    size
        The shape of the kernel over which the median will be calculated.
    axis
        The axes over which to apply the kernel.
        Should either be a scalar or have the same number of items as `size`.
        If :obj:`None` (the default) the kernel spans every axis of the array.
    where
        An optional mask that can be used to exclude parts of the array during
        filtering.
    mode
        The method used to extend the input array beyond its boundaries.
        See :func:`scipy.ndimage.generic_filter` for the definitions.
        Currently, only "mirror", "nearest", "wrap", and "truncate" modes are
        supported.

    Returns
    -------
        A copy of the array with the median filter applied.

    Notes
    -----
    For a kernel with an even number of elements, this function averages the
    two middle elements of the footprint, like :func:`numpy.median`, while
    :func:`scipy.ndimage.median_filter` returns the larger of the two.
    SciPy implements its median filter as a rank filter, selecting the
    element of rank ``size // 2``, so its result is always a value that
    occurs in the footprint and can be stored in an integer array.

    Also like :func:`numpy.median`, the median of a footprint which holds NaN
    is NaN. To ignore NaN instead, exclude it with `where`, as in
    ``where=~np.isnan(array)``.

    Examples
    --------

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

    """
    return ndfilters.generic_filter(
        array=array,
        function=_median,
        size=size,
        axis=axis,
        where=where,
        mode=mode,
    )


@numba.njit(cache=True)
def _median(
    array: np.ndarray,
    args: tuple,
) -> float:

    # As in `trimmed_mean_filter`, the scratch copy of the footprint is
    # partitioned in place, where `np.median` would copy it again.
    n = array.size
    half = n // 2
    select(array, half, 0, n - 1)

    # As with `np.median`, the median of a footprint which holds NaN is NaN.
    # NaN sorts after every number, so it can only be at or after `half`.
    for i in range(half, n):
        if np.isnan(array[i]):
            return np.nan

    if n % 2:
        return float(array[half])

    # The other middle element is the largest of those before `half`, which
    # the selection leaves in no particular order.
    below = array[0]
    for i in range(1, half):
        if array[i] > below:
            below = array[i]

    return float((below + array[half]) / 2)
