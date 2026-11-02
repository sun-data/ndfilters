import numpy as np
import numba

__all__ = [
    "modes",
    "validate_mode",
    "rectify_index_lower",
    "rectify_index_upper",
    "footprint_indices",
]

modes = ("mirror", "nearest", "wrap", "truncate")
"""The boundary modes supported by the filters in this package."""


def validate_mode(mode: str) -> None:
    """
    Raise a :class:`ValueError` if `mode` is not a supported boundary mode.

    The compiled kernels cannot report an unrecognized mode themselves, since
    exceptions raised inside a :func:`numba.prange` loop are not propagated to
    the caller, so `mode` has to be checked before dispatching to them.

    Parameters
    ----------
    mode
        The method used to extend the input array beyond its boundaries.
    """
    if mode not in modes:
        raise ValueError(f"Unrecognized {mode=}, expected one of {modes}.")


@numba.njit(cache=True)
def _rectify_index_mirror(index: int, size: int) -> int:
    """
    Reflect `index` into the interval ``[0, size)`` without repeating the
    elements on the boundary.

    The reflection has a period of ``2 * (size - 1)``, so `index` is folded
    into one period before being reflected. Folding first is what allows an
    index to be rectified even if it is more than one array width outside the
    boundary, which happens whenever the kernel is wider than ``2 * size - 1``.

    Parameters
    ----------
    index
        The out-of-bounds index to rectify.
    size
        The length of the axis being indexed.
    """
    if size == 1:
        return 0
    period = 2 * (size - 1)
    index = index % period
    if index >= size:
        index = period - index
    return index


@numba.njit(cache=True)
def rectify_index_lower(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return _rectify_index_mirror(index, size)
    elif mode == "nearest":
        return 0
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError


@numba.njit(cache=True)
def rectify_index_upper(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return _rectify_index_mirror(index, size)
    elif mode == "nearest":
        return size - 1
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError


@numba.njit(cache=True)
def footprint_indices(
    size_array: int,
    size_kernel: int,
    mode: str,
) -> np.ndarray:
    """
    The index of the element of an axis under each element of the kernel,
    wherever the kernel is centered along the axis.

    The index under element ``k`` of the kernel, centered on element ``i`` of
    the axis, is ``result[i + k]``, since it depends only on their sum.
    The element ``size_kernel // 2`` of the kernel is its center, as in
    :mod:`scipy.ndimage`.
    The indices outside the axis are rectified according to `mode`, or are
    -1 if `mode` is ``"truncate"``, which drops them.
    The filters look the indices up in this table, made before their loops,
    since rectifying them inside the loops, with calls and branches on
    `mode`, made the loops about twice as slow.
    A table with a row for each element of the axis would hold
    ``size_array * size_kernel`` indices, which for a long axis and a wide
    kernel take longer to make than the filter itself.

    Parameters
    ----------
    size_array
        The length of the axis.
    size_kernel
        The length of the kernel along the axis.
    mode
        The method used to extend the array beyond its boundaries.

    Returns
    -------
    :
        The indices, an array of ``size_array + size_kernel - 1`` elements.
    """
    result = np.empty(size_array + size_kernel - 1, dtype=np.int64)
    for i_plus_k in range(result.size):
        j = i_plus_k - size_kernel // 2
        if j < 0:
            if mode == "truncate":
                j = -1
            else:
                j = rectify_index_lower(j, size_array, mode)
        elif j >= size_array:
            if mode == "truncate":
                j = -1
            else:
                j = rectify_index_upper(j, size_array, mode)
        result[i_plus_k] = j
    return result
