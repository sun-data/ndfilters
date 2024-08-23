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
        pytest.param("foo", marks=pytest.mark.xfail),
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
