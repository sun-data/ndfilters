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
            axis_normalized = np.core.numeric.normalize_axis_tuple(
                axis, ndim=array.ndim
            )
        except np.AxisError:
            with pytest.raises(np.AxisError):
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
