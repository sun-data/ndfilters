import numba

__all__ = [
    "rectify_index_lower",
    "rectify_index_upper",
]


@numba.njit(cache=True)
def rectify_index_lower(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return -index
    elif mode == "nearest":
        return 0
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError


@numba.njit(cache=True)
def rectify_index_upper(index: int, size: int, mode: str) -> int:
    if mode == "mirror":
        return ~(index % size + 1)
    elif mode == "nearest":
        return size - 1
    elif mode == "wrap":
        return index % size
    else:  # pragma: nocover
        raise ValueError
