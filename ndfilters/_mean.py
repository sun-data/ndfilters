from typing import Literal
import numpy as np
import numba
import astropy.units as u
from ._indices import footprint_indices
from ._generic import _filter

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
    return _filter(
        array=array,
        kernels=(_mean_filter_1d, _mean_filter_2d, _mean_filter_3d),
        size=size,
        axis=axis,
        where=where,
        mode=mode,
    )


@numba.njit(parallel=True, cache=True)
def _mean_filter_1d(
    array: np.ndarray,
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
) -> np.ndarray:
    result = np.empty_like(array)

    array_shape_t, array_shape_x = array.shape

    (kernel_shape_x,) = size

    index_x = footprint_indices(array_shape_x, kernel_shape_x, mode)

    # Unlike `generic_filter`, which gathers the elements of each kernel
    # footprint into an array for its function, this sums them as it reads
    # them, in double precision even for a single-precision `array`.
    for i in numba.prange(array_shape_t * array_shape_x):

        it = i // array_shape_x
        ix = i % array_shape_x

        total = 0.0
        num_values = 0

        for kx in range(kernel_shape_x):

            jx = index_x[ix, kx]
            if jx < 0:
                continue

            if where[it, jx]:
                total += float(array[it, jx])
                num_values += 1

        if num_values > 0:
            result[it, ix] = total / num_values
        else:
            result[it, ix] = np.nan

    return result


@numba.njit(parallel=True, cache=True)
def _mean_filter_2d(
    array: np.ndarray,
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
) -> np.ndarray:
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y = array.shape

    kernel_shape_x, kernel_shape_y = size

    index_x = footprint_indices(array_shape_x, kernel_shape_x, mode)
    index_y = footprint_indices(array_shape_y, kernel_shape_y, mode)

    for i in numba.prange(array_shape_t * array_shape_x * array_shape_y):

        it = i // (array_shape_x * array_shape_y)
        ix = (i // array_shape_y) % array_shape_x
        iy = i % array_shape_y

        total = 0.0
        num_values = 0

        for kx in range(kernel_shape_x):

            jx = index_x[ix, kx]
            if jx < 0:
                continue

            for ky in range(kernel_shape_y):

                jy = index_y[iy, ky]
                if jy < 0:
                    continue

                if where[it, jx, jy]:
                    total += float(array[it, jx, jy])
                    num_values += 1

        if num_values > 0:
            result[it, ix, iy] = total / num_values
        else:
            result[it, ix, iy] = np.nan

    return result


@numba.njit(parallel=True, cache=True)
def _mean_filter_3d(
    array: np.ndarray,
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
) -> np.ndarray:
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y, array_shape_z = array.shape

    kernel_shape_x, kernel_shape_y, kernel_shape_z = size

    index_x = footprint_indices(array_shape_x, kernel_shape_x, mode)
    index_y = footprint_indices(array_shape_y, kernel_shape_y, mode)
    index_z = footprint_indices(array_shape_z, kernel_shape_z, mode)

    for i in numba.prange(
        array_shape_t * array_shape_x * array_shape_y * array_shape_z
    ):

        it = i // (array_shape_x * array_shape_y * array_shape_z)
        ix = (i // (array_shape_y * array_shape_z)) % array_shape_x
        iy = (i // array_shape_z) % array_shape_y
        iz = i % array_shape_z

        total = 0.0
        num_values = 0

        for kx in range(kernel_shape_x):

            jx = index_x[ix, kx]
            if jx < 0:
                continue

            for ky in range(kernel_shape_y):

                jy = index_y[iy, ky]
                if jy < 0:
                    continue

                for kz in range(kernel_shape_z):

                    jz = index_z[iz, kz]
                    if jz < 0:
                        continue

                    if where[it, jx, jy, jz]:
                        total += float(array[it, jx, jy, jz])
                        num_values += 1

        if num_values > 0:
            result[it, ix, iy, iz] = total / num_values
        else:
            result[it, ix, iy, iz] = np.nan

    return result
