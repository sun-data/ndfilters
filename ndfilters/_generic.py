from typing import Any, Callable, Literal
import numpy as np
import numba
import astropy.units as u
from ._indices import (
    validate_mode,
    footprint_indices,
)

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
        The shape of the kernel over which `function` will be evaluated.
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

    Returns
    -------
        A copy of the array with `function` applied to each kernel footprint.

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
    return _filter(
        array=array,
        kernels=(_generic_filter_1d, _generic_filter_2d, _generic_filter_3d),
        size=size,
        axis=axis,
        where=where,
        mode=mode,
        function=function,
        args=args,
    )


def _filter(
    array: np.ndarray | u.Quantity,
    kernels: tuple[Callable[..., np.ndarray], ...],
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...],
    where: bool | np.ndarray,
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
    **kwargs: Any,
) -> np.ndarray:
    """
    Apply a compiled kernel to every footprint of an array.

    Every filter in this package except the convolution uses this to prepare
    the arguments of its kernel and its result: it checks `mode` and `size`,
    takes the units off `array` and puts them back on the result, promotes an
    integer `array` to floating point, and reshapes `array` and `where` so
    that the axes of the kernel come last, after a single batch axis made of
    the others.

    Parameters
    ----------
    array
        The input array to be filtered.
    kernels
        The compiled kernels for one, two, and three axes, which are called
        with the batched `array` and `where`, `size`, `mode`, and `kwargs`.
    size
        The shape of the kernel.
    axis
        The axes over which to apply the kernel.
        If :obj:`None`, the kernel spans every axis of the array.
    where
        A mask that excludes parts of the array from the filter.
    mode
        The method used to extend the input array beyond its boundaries.
    kwargs
        The other arguments of the kernels.
    """
    validate_mode(mode)

    if isinstance(array, u.Quantity):
        unit = array.unit
        array = array.value
    else:
        unit = None

    # The kernels return a float, and an empty kernel footprint returns NaN,
    # so an integer array has to be promoted before it can hold the result.
    array = np.asanyarray(array)
    array = array.astype(np.result_type(array.dtype, np.float32), copy=False)

    if axis is None:
        axis = tuple(range(array.ndim))
    axis = np.lib.array_utils.normalize_axis_tuple(axis, ndim=array.ndim)

    if isinstance(size, int):
        size = (size,) * len(axis)
    else:
        if len(size) != len(axis):
            raise ValueError(
                f"{size=} should have the same number of elements as {axis=}."
            )

    if any(s < 1 for s in size):
        raise ValueError(f"Every element of {size=} should be a positive integer.")

    axis_numba = ~np.arange(len(axis))[::-1]

    shape = array.shape
    shape_numba = tuple(shape[ax] for ax in axis)

    where = np.broadcast_to(where, shape)

    array_ = np.moveaxis(array, axis, axis_numba)
    where_ = np.moveaxis(where, axis, axis_numba)

    if len(axis) not in (1, 2, 3):  # pragma: nocover
        raise ValueError(f"Only 1-3 axes supported, got {axis=}.")

    kernel = kernels[len(axis) - 1]

    result = kernel(
        array=array_.reshape(-1, *shape_numba),
        size=size,
        where=where_.reshape(-1, *shape_numba),
        mode=mode,
        **kwargs,
    )

    result = result.reshape(array_.shape)
    result = np.moveaxis(result, axis_numba, axis)

    if unit is not None:
        result = result << unit

    return result


@numba.njit(parallel=True, cache=True)
def _generic_filter_1d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
    args: tuple,
) -> np.ndarray:
    result = np.empty_like(array)

    array_shape_t, array_shape_x = array.shape

    (kernel_shape_x,) = size

    index_x = footprint_indices(array_shape_x, kernel_shape_x, mode)

    # Every output element is independent, so the batch axis and the kernel
    # axes are flattened into a single parallel loop. Parallelizing over only
    # the kernel axes starves the thread pool whenever those axes are short.
    for i in numba.prange(array_shape_t * array_shape_x):

        it = i // array_shape_x
        ix = i % array_shape_x

        # The elements of the kernel footprint which are not excluded by
        # `where` are gathered into the front of `values`, and `function` is
        # given a slice of them. Selecting them with a boolean mask instead
        # would build a new array for every output element, which takes
        # longer than a simple function such as the mean itself.
        # Numba hoists the allocation of `values` out of the loop.
        values = np.empty(kernel_shape_x)
        num_values = 0

        for kx in range(kernel_shape_x):

            # Numba checks every signed index for a negative value to wrap
            # around, so the table of indices is read at an unsigned one,
            # which makes the filters up to about 20% faster.
            jx = index_x[np.uint64(ix + kx)]
            if jx < 0:
                continue

            if where[it, jx]:
                values[num_values] = array[it, jx]
                num_values += 1

        if num_values > 0:
            result[it, ix] = function(values[:num_values], args)
        else:
            result[it, ix] = np.nan

    return result


@numba.njit(parallel=True, cache=True)
def _generic_filter_2d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
    args: tuple,
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

        values = np.empty(kernel_shape_x * kernel_shape_y)
        num_values = 0

        for kx in range(kernel_shape_x):

            jx = index_x[np.uint64(ix + kx)]
            if jx < 0:
                continue

            for ky in range(kernel_shape_y):

                jy = index_y[np.uint64(iy + ky)]
                if jy < 0:
                    continue

                if where[it, jx, jy]:
                    values[num_values] = array[it, jx, jy]
                    num_values += 1

        if num_values > 0:
            result[it, ix, iy] = function(values[:num_values], args)
        else:
            result[it, ix, iy] = np.nan

    return result


@numba.njit(parallel=True, cache=True)
def _generic_filter_3d(
    array: np.ndarray,
    function: Callable[[np.ndarray, tuple], float],
    size: tuple[int, ...],
    where: np.ndarray,
    mode: str,
    args: tuple,
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

        values = np.empty(kernel_shape_x * kernel_shape_y * kernel_shape_z)
        num_values = 0

        for kx in range(kernel_shape_x):

            jx = index_x[np.uint64(ix + kx)]
            if jx < 0:
                continue

            for ky in range(kernel_shape_y):

                jy = index_y[np.uint64(iy + ky)]
                if jy < 0:
                    continue

                for kz in range(kernel_shape_z):

                    jz = index_z[np.uint64(iz + kz)]
                    if jz < 0:
                        continue

                    if where[it, jx, jy, jz]:
                        values[num_values] = array[it, jx, jy, jz]
                        num_values += 1

        if num_values > 0:
            result[it, ix, iy, iz] = function(values[:num_values], args)
        else:
            result[it, ix, iy, iz] = np.nan

    return result
