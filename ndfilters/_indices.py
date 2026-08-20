import numba

__all__ = [
    "modes",
    "validate_mode",
    "rectify_index_lower",
    "rectify_index_upper",
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
