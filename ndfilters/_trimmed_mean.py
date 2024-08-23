from typing import Literal
import numpy as np
import numba
import astropy.units as u
import ndfilters

__all__ = [
    "trimmed_mean_filter",
]


def trimmed_mean_filter(
    array: np.ndarray | u.Quantity,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
    proportion: float = 0.25,
) -> np.ndarray:
    """
    Calculate a multidimensional rolling trimmed mean.

    Parameters
    ----------
    array
        The input array to be filtered
    size
        The shape of the kernel over which the trimmed mean will be calculated.
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
    proportion
        The proportion to cut from the top and bottom of the distribution.

    Returns
    -------
        A copy of the array with the trimmed mean filter applied.

    Examples
    --------

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import scipy.datasets
        import ndfilters

        img = scipy.datasets.ascent()
        img_filtered = ndfilters.trimmed_mean_filter(img, size=21)

        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].set_title("original image");
        axs[0].imshow(img, cmap="gray");
        axs[1].set_title("filtered image");
        axs[1].imshow(img_filtered, cmap="gray");

    """
    return ndfilters.generic_filter(
        array=array,
        function=_trimmed_mean,
        size=size,
        axis=axis,
        where=where,
        mode=mode,
        args=(proportion,),
    )


@numba.njit
def _trimmed_mean(
    array: np.ndarray,
    args: tuple[float],
) -> float:

    (proportion,) = args

    nobs = array.size
    if nobs == 0:
        return np.nan
    lowercut = int(proportion * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:  # pragma: nocover
        raise ValueError("Proportion too big.")

    array = np.partition(array, (lowercut, uppercut - 1))

    return np.mean(array[lowercut:uppercut])
