from typing import Callable, Literal
import pytest
import numpy as np
import numba
import scipy.ndimage
import astropy.units as u
import ndfilters


@numba.njit
def _mean(a: np.ndarray, args: tuple = ()) -> float:
    return np.mean(a)


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.random.uniform(size=101),
        np.random.uniform(size=101) * u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _mean,
    ],
)
@pytest.mark.parametrize(
    argnames="size",
    argvalues=[
        5,
        (5,),
    ],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=[
        "mirror",
        "nearest",
        "wrap",
        "truncate",
    ],
)
def test_generic_filter(
    array: np.ndarray | u.Quantity,
    function: Callable[[np.ndarray], float],
    size: int | tuple[int, ...],
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
):
    result = ndfilters.generic_filter(
        array=array,
        function=function,
        size=size,
        mode=mode,
    )
    assert result.shape == array.shape
    assert result.sum() != 0

    if mode != "truncate":
        result_expected = scipy.ndimage.generic_filter(
            input=array,
            function=function,
            size=size,
            mode=mode,
        )

        if isinstance(array, u.Quantity):
            assert np.all(result.value == result_expected)
            assert result.unit == array.unit
        else:
            assert np.all(result == result_expected)


@pytest.mark.parametrize(
    argnames="filter_",
    argvalues=[
        ndfilters.mean_filter,
        ndfilters.median_filter,
        ndfilters.variance_filter,
        ndfilters.trimmed_mean_filter,
    ],
)
def test_generic_filter_mode_invalid(filter_: Callable):
    with pytest.raises(ValueError, match="Unrecognized mode="):
        filter_(np.random.uniform(size=11), size=3, mode="foo")


@pytest.mark.parametrize(
    argnames="size",
    argvalues=[0, -1, (3, 0)],
)
def test_generic_filter_size_invalid(size: int | tuple[int, ...]):
    array = np.random.uniform(size=(11, 12))
    axis = None if not isinstance(size, tuple) else (0, 1)
    with pytest.raises(ValueError, match="should be a positive integer"):
        ndfilters.generic_filter(array, function=_mean, size=size, axis=axis)


@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=[np.int64, np.uint8, np.float32],
)
def test_generic_filter_dtype(dtype: np.dtype):
    """
    The filters return a float, so an integer array is promoted rather than
    truncating the result, or, where the kernel footprint is empty, storing
    NaN in an integer array.
    """
    array = np.arange(25).reshape(5, 5).astype(dtype)

    result = ndfilters.mean_filter(array, size=3, mode="nearest")
    assert np.issubdtype(result.dtype, np.floating)
    assert np.allclose(result[0, 0], np.mean([0, 0, 1, 0, 0, 1, 5, 5, 6]))

    where = np.zeros(array.shape, dtype=bool)
    where[~0, ~0] = True
    result = ndfilters.mean_filter(array, size=3, where=where, mode="nearest")
    assert np.isnan(result[0, 0])
