from typing import Literal
import pytest
import numpy as np
import scipy.ndimage
import scipy.stats
import astropy.units as u
import ndfilters


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.random.random(5),
        np.random.random((5, 6)),
        np.random.random((5, 6, 7)) * u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="size",
    argvalues=[2, (3,), (3, 4), (3, 4, 5)],
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        None,
        0,
        -1,
        (0,),
        (-1,),
        (0, 1),
        (-2, -1),
        (0, 1, 2),
        (2, 1, 0),
    ],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=[
        "mirror",
        "nearest",
        "wrap",
    ],
)
def test_mean_filter(
    array: np.ndarray,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...],
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
):
    kwargs = dict(
        array=array,
        size=size,
        axis=axis,
        mode=mode,
    )

    if axis is None:
        axis_normalized = tuple(range(array.ndim))
    else:
        try:
            axis_normalized = np.lib.array_utils.normalize_axis_tuple(
                axis, ndim=array.ndim
            )
        except np.exceptions.AxisError:
            with pytest.raises(np.exceptions.AxisError):
                ndfilters.mean_filter(**kwargs)
            return

    if isinstance(size, int):
        size_normalized = (size,) * len(axis_normalized)
    else:
        size_normalized = size

    if len(size_normalized) != len(axis_normalized):
        with pytest.raises(ValueError):
            ndfilters.mean_filter(**kwargs)
        return

    result = ndfilters.mean_filter(**kwargs)

    size_scipy = [1] * array.ndim
    for i, ax in enumerate(axis_normalized):
        size_scipy[ax] = size_normalized[i]

    expected = scipy.ndimage.uniform_filter(
        input=array,
        size=size_scipy,
        mode=mode,
    )

    if isinstance(result, u.Quantity):
        assert np.allclose(result.value, expected)
        assert result.unit == array.unit
    else:
        assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="size_array",
    argvalues=[1, 2, 5],
)
@pytest.mark.parametrize(
    argnames="size",
    argvalues=[3, 5, 9, 11, 13, 21],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=["mirror", "nearest", "wrap"],
)
def test_mean_filter_kernel_wider_than_array(
    size_array: int,
    size: int,
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
):
    """
    A kernel wider than ``2 * size_array - 1`` places an index more than one
    array width outside the boundary, which used to be rectified to an
    out-of-bounds index in "mirror" mode.
    """
    array = np.random.random(size_array)

    result = ndfilters.mean_filter(array, size=size, mode=mode)

    expected = scipy.ndimage.uniform_filter(array, size=size, mode=mode)

    assert np.allclose(result, expected)
