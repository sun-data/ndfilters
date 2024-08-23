from typing import Callable, Literal
import numpy as np
import numba
import astropy.units as u

__all__ = [
    "generic_filter",
]


def generic_filter(
    array: np.ndarray | u.Quantity,
    function: Callable[[np.ndarray, tuple], float],
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...] = None,
    where: bool | np.ndarray = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
    args: tuple = (),
) -> np.ndarray:
    """
    Filter a multidimensional array using an arbitrary compiled function.

    Parameters
    ----------
    array
        The input array to be filtered
    function
        The function to applied to each kernel footprint.
        This is usually either a Numpy reduction function like :func:`numpy.mean`,
        or a function compiled using :func:`numba.njit`.
        This function must accept a 1D array and a tuple of extra arguments as
        input and return a scalar.
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
    args
        Extra arguments to pass to function.

    Examples
    --------

    .. jupyter-execute::

        import numpy as np
        import numba
        import matplotlib.pyplot as plt
        import scipy.datasets
        import ndfilters

        # Download a sample image
        img = scipy.datasets.ascent()

        # Define a compiled function to apply at every
        # kernel footprint.
        @numba.njit
        def function(a: np.ndarray, args: tuple) -> float:
            return np.mean(a)

        # Filter the image using an arbitrary function.
        img_filtered = ndfilters.generic_filter(
            function=function,
            array=img,
            size=21,
        )

        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].set_title("original image");
        axs[0].imshow(img, cmap="gray");
        axs[1].set_title("filtered image");
        axs[1].imshow(img_filtered, cmap="gray");

    """
    if isinstance(array, u.Quantity):
        unit = array.unit
        array = array.value
    else:
        unit = None

    if axis is None:
        axis = tuple(range(array.ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=array.ndim)

    if isinstance(size, int):
        size = (size,) * len(axis)
    else:
        if len(size) != len(axis):
            raise ValueError(
                f"{size=} should have the same number of elements as {axis=}."
            )

    axis_numba = ~np.arange(len(axis))[::-1]

    shape = array.shape
    shape_numba = tuple(shape[ax] for ax in axis)

    where = np.broadcast_to(where, shape)

    array_ = np.moveaxis(array, axis, axis_numba)
    where_ = np.moveaxis(where, axis, axis_numba)

    if len(axis) == 1:
        _generic_filter_nd = _generic_filter_1d
    elif len(axis) == 2:
        _generic_filter_nd = _generic_filter_2d
    elif len(axis) == 3:
        _generic_filter_nd = _generic_filter_3d
    else:  # pragma: nocover
        raise ValueError(f"Only 1-3 axes supported, got {axis=}.")

    result = _generic_filter_nd(
        array=array_.reshape(-1, *shape_numba),
        function=function,
        size=size,
        where=where_.reshape(-1, *shape_numba),
        mode=mode,
        args=args,
    )

    result = result.reshape(array_.shape)
    result = np.moveaxis(result, axis_numba, axis)

    if unit is not None:
        result = result << unit

    return result


@numba.njit
def _rectify_index_lower(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return -index
    elif mode == "nearest":
        return 0
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError


@numba.njit
def _rectify_index_upper(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return ~(index % size + 1)
    elif mode == "nearest":
        return size - 1
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError


@numba.njit(parallel=True)
def _generic_filter_1d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int],
    where: np.ndarray,
    mode: str,
    args: tuple,
):
    result = np.empty_like(array)

    array_shape_t, array_shape_x = array.shape

    (kernel_shape_x,) = size

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):

            values = np.zeros(shape=size)
            mask = np.zeros(shape=size, dtype=np.bool_)

            for kx in range(kernel_shape_x):

                px = kx - kernel_shape_x // 2
                jx = ix + px

                if jx < 0:
                    if mode == "truncate":
                        continue
                    jx = _rectify_index_lower(jx, array_shape_x, mode)
                elif jx >= array_shape_x:
                    if mode == "truncate":
                        continue
                    jx = _rectify_index_upper(jx, array_shape_x, mode)

                values[kx] = array[it, jx]
                mask[kx] = where[it, jx]

            result[it, ix] = function(values[mask], args)

    return result


@numba.njit(parallel=True)
def _generic_filter_2d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int, int],
    where: np.ndarray,
    mode: str,
    args: tuple,
):
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y = array.shape

    kernel_shape_x, kernel_shape_y = size

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):
            for iy in numba.prange(array_shape_y):

                values = np.zeros(shape=size)
                mask = np.zeros(shape=size, dtype=np.bool_)

                for kx in range(kernel_shape_x):

                    px = kx - kernel_shape_x // 2
                    jx = ix + px

                    if jx < 0:
                        if mode == "truncate":
                            continue
                        jx = _rectify_index_lower(jx, array_shape_x, mode)
                    elif jx >= array_shape_x:
                        if mode == "truncate":
                            continue
                        jx = _rectify_index_upper(jx, array_shape_x, mode)

                    for ky in range(kernel_shape_y):

                        py = ky - kernel_shape_y // 2
                        jy = iy + py

                        if jy < 0:
                            if mode == "truncate":
                                continue
                            jy = _rectify_index_lower(jy, array_shape_y, mode)
                        elif jy >= array_shape_y:
                            if mode == "truncate":
                                continue
                            jy = _rectify_index_upper(jy, array_shape_y, mode)

                        values[kx, ky] = array[it, jx, jy]
                        mask[kx, ky] = where[it, jx, jy]

                values = values.reshape(-1)
                mask = mask.reshape(-1)

                result[it, ix, iy] = function(values[mask], args)

    return result


@numba.njit(parallel=True)
def _generic_filter_3d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int, int, int],
    where: np.ndarray,
    mode: str,
    args: tuple,
):
    result = np.empty_like(array)

    array_shape_t, array_shape_x, array_shape_y, array_shape_z = array.shape

    kernel_shape_x, kernel_shape_y, kernel_shape_z = size

    for it in range(array_shape_t):

        for ix in numba.prange(array_shape_x):
            for iy in numba.prange(array_shape_y):
                for iz in numba.prange(array_shape_z):

                    values = np.zeros(shape=size)
                    mask = np.zeros(shape=size, dtype=np.bool_)

                    for kx in range(kernel_shape_x):

                        px = kx - kernel_shape_x // 2
                        jx = ix + px

                        if jx < 0:
                            if mode == "truncate":
                                continue
                            jx = _rectify_index_lower(jx, array_shape_x, mode)
                        elif jx >= array_shape_x:
                            if mode == "truncate":
                                continue
                            jx = _rectify_index_upper(jx, array_shape_x, mode)

                        for ky in range(kernel_shape_y):

                            py = ky - kernel_shape_y // 2
                            jy = iy + py

                            if jy < 0:
                                if mode == "truncate":
                                    continue
                                jy = _rectify_index_lower(jy, array_shape_y, mode)
                            elif jy >= array_shape_y:
                                if mode == "truncate":
                                    continue
                                jy = _rectify_index_upper(jy, array_shape_y, mode)

                            for kz in range(kernel_shape_z):

                                pz = kz - kernel_shape_z // 2
                                jz = iz + pz

                                if jz < 0:
                                    if mode == "truncate":
                                        continue
                                    jz = _rectify_index_lower(jz, array_shape_z, mode)
                                elif jz >= array_shape_z:
                                    if mode == "truncate":
                                        continue
                                    jz = _rectify_index_upper(jz, array_shape_z, mode)

                                values[kx, ky, kz] = array[it, jx, jy, jz]
                                mask[kx, ky, kz] = where[it, jx, jy, jz]

                    values = values.reshape(-1)
                    mask = mask.reshape(-1)

                    result[it, ix, iy, iz] = function(values[mask], args)

    return result
