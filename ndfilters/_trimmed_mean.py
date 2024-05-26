import numpy as np
import numba

__all__ = [
    "trimmed_mean_filter",
]


def trimmed_mean_filter(
    array: np.ndarray,
    kernel_shape: int | tuple[int, ...],
    proportion: float = 0.25,
    axis: None | int | tuple[int, ...] = None,
) -> np.ndarray:
    """
    Calculate a multidimensional rolling trimmed mean.

    Parameters
    ----------
    array
        The input array to be filtered
    kernel_shape
        The shape of the kernel over which the trimmed mean will be calculated.
    proportion
        The proportion to cut from the top and bottom of the distribution.
    axis
        The axes over which to apply the kernel. If :class:`None` the kernel
        is applied to every axis.

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
        img_filtered = ndfilters.trimmed_mean_filter(img, kernel_shape=21)

        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].set_title("original image");
        axs[0].imshow(img, cmap="gray");
        axs[1].set_title("filtered image");
        axs[1].imshow(img_filtered, cmap="gray");

    """

    if axis is None:
        axis = tuple(range(array.ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=array.ndim)

    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * len(axis)
    else:
        if len(kernel_shape) != len(axis):
            raise ValueError(
                f"`kernel_shape` should have the same number of elements, "
                f"{len(kernel_shape)}, as `axis`, {len(axis)}"
            )
    kernel_shape = tuple(np.array(kernel_shape)[np.argsort(axis)])

    shape_orthogonal = tuple(
        1 if ax in axis else array.shape[ax] for ax in range(array.ndim)
    )

    result = np.zeros_like(array)

    for index in np.ndindex(*shape_orthogonal):

        index = list(index)
        for ax in axis:
            index[ax] = slice(None)
        index = tuple(index)

        if len(axis) == 1:
            result[index] = _mean_trimmed_1d(
                array=array[index],
                kernel_shape=kernel_shape,
                proportion=proportion,
            )
        elif len(axis) == 2:
            result[index] = _mean_trimmed_2d(
                array=array[index],
                kernel_shape=kernel_shape,
                proportion=proportion,
            )
        elif len(axis) == 3:
            result[index] = _mean_trimmed_3d(
                array=array[index],
                kernel_shape=kernel_shape,
                proportion=proportion,
            )
        else:
            raise ValueError(
                "Too many axis parameters, only 1-3 reduction axes are supported."
            )

    return result


@numba.njit(parallel=True)
def _mean_trimmed_1d(
    array: np.ndarray,
    kernel_shape: tuple[int],
    proportion: float,
):
    result = np.empty_like(array)

    (array_shape_x,) = array.shape

    (kernel_shape_x,) = kernel_shape
    kernel_size = kernel_shape_x

    for ix in numba.prange(array_shape_x):

        values = np.empty(kernel_size)
        for kx in range(kernel_shape_x):
            px = kx - kernel_shape_x // 2
            jx = ix + px

            if jx < 0:
                jx = -jx
            elif jx >= array_shape_x:
                jx = ~(jx % array_shape_x + 1)

            values[kx] = array[jx,]

        lowercut = int(proportion * kernel_size)
        uppercut = kernel_size - lowercut

        for i in range(lowercut):
            j_min = i
            for j in range(i + 1, kernel_size):
                if values[j] < values[j_min]:
                    j_min = j
            if j_min != i:
                values[i], values[j_min] = values[j_min], values[i]

        for i in range(uppercut + 1):
            j_max = i
            for j in range(i + 1, kernel_size - lowercut):
                if values[~j] > values[~j_max]:
                    j_max = j
            if j_max != i:
                values[~i], values[~j_max] = values[~j_max], values[~i]

        sum_values = 0
        num_values = 0
        for i in range(lowercut, uppercut):
            sum_values += values[i]
            num_values += 1

        result[ix,] = sum_values / num_values

    return result


@numba.njit(parallel=True)
def _mean_trimmed_2d(
    array: np.ndarray,
    kernel_shape: tuple[int, int],
    proportion: float,
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y = array.shape

    kernel_shape_x, kernel_shape_y = kernel_shape
    kernel_size = kernel_shape_x * kernel_shape_y

    for ix in numba.prange(array_shape_x):
        for iy in numba.prange(array_shape_y):

            values = np.empty(kernel_size)
            for kx in range(kernel_shape_x):
                for ky in range(kernel_shape_y):

                    px = kx - kernel_shape_x // 2
                    py = ky - kernel_shape_y // 2

                    jx = ix + px
                    jy = iy + py

                    if jx < 0:
                        jx = -jx
                    elif jx >= array_shape_x:
                        jx = ~(jx % array_shape_x + 1)

                    if jy < 0:
                        jy = -jy
                    elif jy >= array_shape_y:
                        jy = ~(jy % array_shape_y + 1)

                    index_flat = kx * kernel_shape_y + ky
                    values[index_flat] = array[jx, jy]

            lowercut = int(proportion * kernel_size)
            uppercut = kernel_size - lowercut

            for i in range(lowercut):
                j_min = i
                for j in range(i + 1, kernel_size):
                    if values[j] < values[j_min]:
                        j_min = j
                if j_min != i:
                    values[i], values[j_min] = values[j_min], values[i]

            for i in range(uppercut + 1):
                j_max = i
                for j in range(i + 1, kernel_size - lowercut):
                    if values[~j] > values[~j_max]:
                        j_max = j
                if j_max != i:
                    values[~i], values[~j_max] = values[~j_max], values[~i]

            sum_values = 0
            num_values = 0
            for i in range(lowercut, uppercut):
                sum_values += values[i]
                num_values += 1

            result[ix, iy] = sum_values / num_values

    return result


@numba.njit(parallel=True)
def _mean_trimmed_3d(
    array: np.ndarray,
    kernel_shape: tuple[int, int, int],
    proportion: float,
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y, array_shape_z = array.shape

    kernel_shape_x, kernel_shape_y, kernel_shape_z = kernel_shape
    kernel_size = kernel_shape_x * kernel_shape_y * kernel_shape_z

    for ix in numba.prange(array_shape_x):
        for iy in numba.prange(array_shape_y):
            for iz in numba.prange(array_shape_z):

                values = np.empty(kernel_size)
                for kx in range(kernel_shape_x):
                    for ky in range(kernel_shape_y):
                        for kz in range(kernel_shape_z):

                            px = kx - kernel_shape_x // 2
                            py = ky - kernel_shape_y // 2
                            pz = kz - kernel_shape_z // 2

                            jx = ix + px
                            jy = iy + py
                            jz = iz + pz

                            if jx < 0:
                                jx = -jx
                            elif jx >= array_shape_x:
                                jx = ~(jx % array_shape_x + 1)

                            if jy < 0:
                                jy = -jy
                            elif jy >= array_shape_y:
                                jy = ~(jy % array_shape_y + 1)

                            if jz < 0:
                                jz = -jz
                            elif jz >= array_shape_z:
                                jz = ~(jz % array_shape_z + 1)

                            index_flat = kx * kernel_shape_y + ky
                            index_flat = index_flat * kernel_shape_z + kz

                            values[index_flat] = array[jx, jy, jz]

                lowercut = int(proportion * kernel_size)
                uppercut = kernel_size - lowercut

                for i in range(lowercut):
                    j_min = i
                    for j in range(i + 1, kernel_size):
                        if values[j] < values[j_min]:
                            j_min = j
                    if j_min != i:
                        values[i], values[j_min] = values[j_min], values[i]

                for i in range(uppercut + 1):
                    j_max = i
                    for j in range(i + 1, kernel_size - lowercut):
                        if values[~j] > values[~j_max]:
                            j_max = j
                    if j_max != i:
                        values[~i], values[~j_max] = values[~j_max], values[~i]

                sum_values = 0
                num_values = 0
                for i in range(lowercut, uppercut):
                    sum_values += values[i]
                    num_values += 1

                result[ix, iy, iz] = sum_values / num_values

    return result
