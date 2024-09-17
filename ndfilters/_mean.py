from typing import Literal
import numpy as np
import numba
import astropy.units as u
import ndfilters

__all__ = [
    "mean_filter",
]


def mean_filter(
    array: np.ndarray | u.Quantity,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
) -> np.ndarray:
    """
    Calculate a multidimensional rolling mean.

    Parameters
    ----------
    array
        The input array to be filtered
    size
        The shape of the kernel over which the mean will be calculated.
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
        A copy of the array with the mean filter applied.

    Examples
    --------

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import scipy.datasets
        import ndfilters

        img = scipy.datasets.ascent()
        img_filtered = ndfilters.mean_filter(img, size=21)

        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].set_title("original image");
        axs[0].imshow(img, cmap="gray");
        axs[1].set_title("filtered image");
        axs[1].imshow(img_filtered, cmap="gray");

    """
    return ndfilters.generic_filter(
        array=array,
        function=_mean,
        size=size,
        axis=axis,
        where=where,
        mode=mode,
    )


@numba.njit
def _mean(
    array: np.ndarray,
    args: tuple[float],
) -> float:
    return np.mean(array)
