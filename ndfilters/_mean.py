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

            # As in `generic_filter`, the table of indices is read at an
            # unsigned index, which numba does not check for a negative value.
            jx = index_x[np.uint64(ix + kx)]
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

    # The sum of the elements of a footprint under `where`, and their number,
    # are separable: they are the sums along y of the sums along x.
    # So each tile of a row of the result is made from the sums along x of the
    # elements under the kernel wherever it is centered in the tile, which are
    # then summed along y, in `kernel_shape_x + kernel_shape_y` additions for
    # each element instead of their product.
    # The sums along x are kept in a buffer of about the size of a tile, which
    # numba allocates once for each thread, since a buffer as large as the
    # array, made on every call, made the filter slower, not faster, on many
    # threads, which wait for the pages of new memory to be mapped.
    tile_y = min(array_shape_y, 4096)
    num_tiles_y = (array_shape_y + tile_y - 1) // tile_y

    for i in numba.prange(array_shape_t * array_shape_x * num_tiles_y):

        it = i // (array_shape_x * num_tiles_y)
        ix = (i // num_tiles_y) % array_shape_x
        iy0 = (i % num_tiles_y) * tile_y

        num_y = min(tile_y, array_shape_y - iy0)
        halo_y = num_y + kernel_shape_y - 1

        # Numba hoists the allocation of the buffers out of the loop, making
        # one for each thread, only if their size is the same for every tile,
        # so they are sized for a whole tile, not by `halo_y`.
        # Element `hy` of the buffers is the sum along x at `index_y[iy0 + hy]`.
        total_x = np.empty(tile_y + kernel_shape_y - 1)
        count_x = np.empty(tile_y + kernel_shape_y - 1, dtype=np.int64)
        total_x[:halo_y] = 0.0
        count_x[:halo_y] = 0

        for kx in range(kernel_shape_x):

            jx = index_x[np.uint64(ix + kx)]
            if jx < 0:
                continue

            for hy in range(halo_y):

                jy = index_y[np.uint64(iy0 + hy)]
                if jy < 0:
                    continue

                if where[it, jx, jy]:
                    total_x[hy] += float(array[it, jx, jy])
                    count_x[hy] += 1

        for ny in range(num_y):

            total = 0.0
            num_values = 0
            for ky in range(kernel_shape_y):
                total += total_x[ny + ky]
                num_values += count_x[ny + ky]

            if num_values > 0:
                result[it, ix, iy0 + ny] = total / num_values
            else:
                result[it, ix, iy0 + ny] = np.nan

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

    # As in the 2D kernel, but each tile of a plane of the result is made from
    # the sums along x, which are summed along y, one row of the tile at a
    # time, and then along z.
    # The buffer of the sums along x holds a tile and its halo along both y
    # and z, so for a kernel which is wide along both, it grows with the
    # product of its sizes along them.
    tile_z = min(array_shape_z, 1024)
    tile_y = min(array_shape_y, 8192 // tile_z)
    num_tiles_y = (array_shape_y + tile_y - 1) // tile_y
    num_tiles_z = (array_shape_z + tile_z - 1) // tile_z
    num_tiles = num_tiles_y * num_tiles_z

    for i in numba.prange(array_shape_t * array_shape_x * num_tiles):

        it = i // (array_shape_x * num_tiles)
        ix = (i // num_tiles) % array_shape_x
        iy0 = ((i % num_tiles) // num_tiles_z) * tile_y
        iz0 = (i % num_tiles_z) * tile_z

        num_y = min(tile_y, array_shape_y - iy0)
        num_z = min(tile_z, array_shape_z - iz0)
        halo_y = num_y + kernel_shape_y - 1
        halo_z = num_z + kernel_shape_z - 1

        # As in the 2D kernel, the buffers are sized for a whole tile, so that
        # numba hoists their allocation out of the loop.
        total_x = np.empty((tile_y + kernel_shape_y - 1, tile_z + kernel_shape_z - 1))
        count_x = np.empty(
            (tile_y + kernel_shape_y - 1, tile_z + kernel_shape_z - 1),
            dtype=np.int64,
        )
        total_x[:halo_y, :halo_z] = 0.0
        count_x[:halo_y, :halo_z] = 0

        for kx in range(kernel_shape_x):

            jx = index_x[np.uint64(ix + kx)]
            if jx < 0:
                continue

            for hy in range(halo_y):

                jy = index_y[np.uint64(iy0 + hy)]
                if jy < 0:
                    continue

                for hz in range(halo_z):

                    jz = index_z[np.uint64(iz0 + hz)]
                    if jz < 0:
                        continue

                    if where[it, jx, jy, jz]:
                        total_x[hy, hz] += float(array[it, jx, jy, jz])
                        count_x[hy, hz] += 1

        total_y = np.empty(tile_z + kernel_shape_z - 1)
        count_y = np.empty(tile_z + kernel_shape_z - 1, dtype=np.int64)

        for ny in range(num_y):

            total_y[:halo_z] = 0.0
            count_y[:halo_z] = 0

            for ky in range(kernel_shape_y):
                for hz in range(halo_z):
                    total_y[hz] += total_x[ny + ky, hz]
                    count_y[hz] += count_x[ny + ky, hz]

            for nz in range(num_z):

                total = 0.0
                num_values = 0
                for kz in range(kernel_shape_z):
                    total += total_y[nz + kz]
                    num_values += count_y[nz + kz]

                if num_values > 0:
                    result[it, ix, iy0 + ny, iz0 + nz] = total / num_values
                else:
                    result[it, ix, iy0 + ny, iz0 + nz] = np.nan

    return result
