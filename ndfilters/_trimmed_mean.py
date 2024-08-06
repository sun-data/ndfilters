from typing import Literal, Callable
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
    mode: Literal["mirror"] = "mirror",
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
        Currently, only "reflect" mode is supported.
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

    kernel_size = array.size

    lowercut = int(proportion * kernel_size)
    uppercut = kernel_size - lowercut

    for i in range(lowercut):
        j_min = i
        for j in range(i + 1, kernel_size):
            if array[j] < array[j_min]:
                j_min = j
        if j_min != i:
            array[i], array[j_min] = array[j_min], array[i]

    for i in range(uppercut + 1):
        j_max = i
        for j in range(i + 1, kernel_size - lowercut):
            if array[~j] > array[~j_max]:
                j_max = j
        if j_max != i:
            array[~i], array[~j_max] = array[~j_max], array[~i]

    sum_values = 0
    num_values = 0
    for i in range(lowercut, uppercut):
        sum_values += array[i]
        num_values += 1

    return sum_values / num_values
