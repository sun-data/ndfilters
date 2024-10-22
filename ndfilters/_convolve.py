from typing import Literal
import numpy as np
import numba
import astropy.units as u
from ._indices import (
    rectify_index_lower,
    rectify_index_upper,
)

__all__ = [
    "convolve",
]


def convolve(
    array: np.ndarray | u.Quantity,
    kernel: np.ndarray | u.Quantity,
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
) -> np.ndarray:
    """
    Multidimensional convolution of an array with a given kernel.

    This function differs from :func:`scipy.ndimage.convolve` and
    :func:`astropy.convolution.convolve` because it implements a vectorized
    convolution operation where the kernel is allowed to vary along axes
    orthogonal to the convolution axes.

    Parameters
    ----------
    array
        The input array to be convolved.
    kernel
        The convolution kernel.
        Any non-convolution axes must be broadcastable with `array`.
    axis
        The axes of `array` over which to apply the kernel.
        If :obj:`None`, it is assumed that the convolution is applied to all
        the axes of `array`.
    where
        An optional mask that can be used to exclude elements of `array`
        during the convolution.
    mode
        The method used to extend `array` beyond its boundaries.
    """
    if isinstance(array, u.Quantity):
        unit = array.unit
        array = array.value
    else:
        unit = None

    if axis is None:
        axis = tuple(range(array.ndim))
    axis = np.array(axis)
    axis = np.core.numeric.normalize_axis_tuple(~axis, ndim=array.ndim)
    axis = ~np.array(axis)

    shape_kernel = list(kernel.shape)
    for ax in axis:
        shape_kernel[ax] = 1

    shape = np.broadcast_shapes(array.shape, shape_kernel, np.shape(where))

    shape_kernel = list(shape)
    for ax in axis:
        shape_kernel[ax] = kernel.shape[ax]

    array = np.broadcast_to(array, shape)
    kernel = np.broadcast_to(kernel, shape_kernel)
    where = np.broadcast_to(where, shape)

    axis_numba = ~np.arange(len(axis))[::-1]
    shape_numba = tuple(shape[ax] for ax in axis)
    shape_kernel_numba = tuple(shape_kernel[ax] for ax in axis)

    array_ = np.moveaxis(array, axis, axis_numba)
    kernel_ = np.moveaxis(kernel, axis, axis_numba)
    where_ = np.moveaxis(where, axis, axis_numba)

    if len(axis) == 1:
        _convolve_nd = _convolve_1d
    elif len(axis) == 2:
        _convolve_nd = _convolve_2d
    elif len(axis) == 3:
        _convolve_nd = _convolve_3d
    else:  # pragma: nocover
        raise ValueError(f"Only 1-3 axes supported, got {axis=}.")

    result = _convolve_nd(
        array=array_.reshape(-1, *shape_numba),
        kernel=kernel_.reshape(-1, *shape_kernel_numba),
        where=where_.reshape(-1, *shape_numba),
        mode=mode,
    )

    result = result.reshape(array_.shape)
    result = np.moveaxis(result, axis_numba, axis)

    if unit is not None:
        result = result << unit

    return result


@numba.njit(parallel=True)
def _convolve_1d(
    array: np.ndarray,
    kernel: np.ndarray,
    where: np.ndarray,
    mode: str,
):
    result = np.zeros_like(array)

    array_shape_t, array_shape_x = array.shape

    _, kernel_shape_x = kernel.shape

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):

            r = 0

            for kx in range(kernel_shape_x):

                px = kx - (kernel_shape_x - 1) // 2
                jx = ix + px

                if jx < 0:
                    if mode == "truncate":
                        continue
                    jx = rectify_index_lower(jx, array_shape_x, mode)
                elif jx >= array_shape_x:
                    if mode == "truncate":
                        continue
                    jx = rectify_index_upper(jx, array_shape_x, mode)

                if where[it, jx]:
                    array_tx = array[it, jx]
                    kernel_tx = kernel[it, ~kx]
                    r += array_tx * kernel_tx

            result[it, ix] = r

    return result


@numba.njit(parallel=True)
def _convolve_2d(
    array: np.ndarray,
    kernel: np.ndarray,
    where: np.ndarray,
    mode: str,
):
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y = array.shape

    _, kernel_shape_x, kernel_shape_y = kernel.shape

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):
            for iy in numba.prange(array_shape_y):

                r = 0

                for kx in range(kernel_shape_x):

                    px = kx - (kernel_shape_x - 1) // 2
                    jx = ix + px

                    if jx < 0:
                        if mode == "truncate":
                            continue
                        jx = rectify_index_lower(jx, array_shape_x, mode)
                    elif jx >= array_shape_x:
                        if mode == "truncate":
                            continue
                        jx = rectify_index_upper(jx, array_shape_x, mode)

                    for ky in range(kernel_shape_y):

                        py = ky - (kernel_shape_y - 1) // 2
                        jy = iy + py

                        if jy < 0:
                            if mode == "truncate":
                                continue
                            jy = rectify_index_lower(jy, array_shape_y, mode)
                        elif jy >= array_shape_y:
                            if mode == "truncate":
                                continue
                            jy = rectify_index_upper(jy, array_shape_y, mode)

                        if where[it, jx, jy]:
                            array_txy = array[it, jx, jy]
                            kernel_txy = kernel[it, ~kx, ~ky]
                            r += array_txy * kernel_txy

                result[it, ix, iy] = r

    return result


@numba.njit(parallel=True)
def _convolve_3d(
    array: np.ndarray,
    kernel: np.ndarray,
    where: np.ndarray,
    mode: str,
):
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y, array_shape_z = array.shape

    _, kernel_shape_x, kernel_shape_y, kernel_shape_z = kernel.shape

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):
            for iy in numba.prange(array_shape_y):
                for iz in numba.prange(array_shape_z):

                    r = 0

                    for kx in range(kernel_shape_x):

                        px = kx - (kernel_shape_x - 1) // 2
                        jx = ix + px

                        if jx < 0:
                            if mode == "truncate":
                                continue
                            jx = rectify_index_lower(jx, array_shape_x, mode)
                        elif jx >= array_shape_x:
                            if mode == "truncate":
                                continue
                            jx = rectify_index_upper(jx, array_shape_x, mode)

                        for ky in range(kernel_shape_y):

                            py = ky - (kernel_shape_y - 1) // 2
                            jy = iy + py

                            if jy < 0:
                                if mode == "truncate":
                                    continue
                                jy = rectify_index_lower(jy, array_shape_y, mode)
                            elif jy >= array_shape_y:
                                if mode == "truncate":
                                    continue
                                jy = rectify_index_upper(jy, array_shape_y, mode)

                            for kz in range(kernel_shape_z):

                                pz = kz - (kernel_shape_z - 1) // 2
                                jz = iz + pz

                                if jz < 0:
                                    if mode == "truncate":
                                        continue
                                    jz = rectify_index_lower(jz, array_shape_z, mode)
                                elif jz >= array_shape_z:
                                    if mode == "truncate":
                                        continue
                                    jz = rectify_index_upper(jz, array_shape_z, mode)

                                if where[it, jx, jy, jz]:
                                    array_txyz = array[it, jx, jy, jz]
                                    kernel_txyz = kernel[it, ~kx, ~ky, ~kz]
                                    r += array_txyz * kernel_txyz

                    result[it, ix, iy, iz] = r

    return result
