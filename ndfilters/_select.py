import numpy as np
import numba

__all__ = [
    "sorts_before",
    "select",
]


@numba.njit(cache=True)
def sorts_before(a: float, b: float) -> bool:
    """
    Whether `a` sorts before `b`, with NaN after every number, as in
    :func:`numpy.sort`.

    Parameters
    ----------
    a
        The first number.
    b
        The second number.
    """
    return a < b or (np.isnan(b) and not np.isnan(a))


@numba.njit(cache=True)
def select(a: np.ndarray, k: int, low: int, high: int) -> None:
    """
    Partially sort the elements ``low`` through ``high`` of `a` in place, so
    that element `k` is the one which would be there if they were sorted,
    with none of those before it sorting after it and none of those after it
    sorting before it.

    This is the selection which :func:`numpy.partition` makes, with NaN
    sorting after every number, but in place.
    The filters use it on their footprints instead of :func:`numpy.partition`
    and :func:`numpy.median`, for which numba allocates a copy of the array
    on every call, and :func:`numpy.partition` several more arrays besides,
    which take far longer than the selection itself for the few elements of
    a footprint.

    Parameters
    ----------
    a
        The array to partially sort.
    k
        The index of the element to select, from ``low`` to ``high``.
    low
        The index of the first element to sort.
    high
        The index of the last element to sort.
    """
    while low < high:

        # The pivot is the median of the first, middle, and last elements,
        # which are sorted, so that the first does not sort after the pivot
        # and the last does not sort before it, which stops the scans below
        # before they leave the interval.
        mid = (low + high) // 2
        if sorts_before(a[mid], a[low]):
            a[low], a[mid] = a[mid], a[low]
        if sorts_before(a[high], a[mid]):
            a[mid], a[high] = a[high], a[mid]
            if sorts_before(a[mid], a[low]):
                a[low], a[mid] = a[mid], a[low]
        pivot = a[mid]

        # Hoare's partition, whose scans stop on the elements equal to the
        # pivot, so that an interval of equal elements is split in the middle
        # instead of one at a time.
        i = low
        j = high
        while True:
            i += 1
            while sorts_before(a[i], pivot):
                i += 1
            j -= 1
            while sorts_before(pivot, a[j]):
                j -= 1
            if i >= j:
                break
            a[i], a[j] = a[j], a[i]

        # Now no element from `low` through `j` sorts after the pivot, and
        # none after `j` sorts before it.
        if k <= j:
            high = j
        else:
            low = j + 1
