import numpy as np
import numba

__all__ = [
    "mean_filter",
]


def mean_filter(
    array: np.ndarray,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
) -> np.ndarray:
    """
    Calculate a multidimensional rolling trimmed mean.
    The kernel is truncated at the edges of the array.

    Parameters
    ----------
    array
        The input array to be filtered
    size
        The shape of the kernel over which the trimmed mean will be calculated.
    axis
        The axes over which to apply the kernel. If :class:`None` the kernel
        is applied to every axis.
    where
        A boolean mask used to select which elements of the input array to filter.

    Returns
    -------
        A copy of the array with a mean filter applied.

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
        axs[1].set_title("mean filtered image");
        axs[1].imshow(img_filtered, cmap="gray");
    """
    array, where = np.broadcast_arrays(array, where, subok=True)

    axis = np.core.numeric.normalize_axis_tuple(axis=axis, ndim=array.ndim)

    if isinstance(size, int):
        size = (size,) * len(axis)

    result = array
    for sz, ax in zip(size, axis, strict=True):
        result = _mean_filter_1d(
            array=result,
            size=sz,
            axis=ax,
            where=where,
        )

    return result


def _mean_filter_1d(
    array: np.ndarray,
    size: int,
    axis: int,
    where: np.ndarray,
) -> np.ndarray:

    array = np.moveaxis(array, axis, ~0)
    where = np.moveaxis(where, axis, ~0)

    shape = array.shape

    array = array.reshape(-1, shape[~0])
    where = where.reshape(-1, shape[~0])

    result = _mean_filter_1d_numba(
        array=array,
        size=size,
        where=where,
    )

    result = result.reshape(shape)

    result = np.moveaxis(result, ~0, axis)

    return result


@numba.njit(parallel=True)
def _mean_filter_1d_numba(
    array: np.ndarray,
    size: int,
    where: np.ndarray,
) -> np.ndarray:

    result = np.empty_like(array)
    num_t, num_x = array.shape

    halfsize = size // 2

    for t in numba.prange(num_t):

        for i in range(num_x):

            sum = 0
            count = 0
            for j in range(size):

                j2 = j - halfsize

                k = i + j2
                if k < 0:
                    continue
                elif k >= num_x:
                    continue

                if where[t, k]:
                    sum += array[t, k]
                    count += 1

            result[t, i] = sum / count

    return result
