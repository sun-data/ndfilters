from typing import Literal
import pytest
import numpy as np
import scipy.ndimage
import scipy.stats
import ndfilters


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.random.random(5),
        np.random.random((5, 6)),
        np.random.random((5, 6, 7)),
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
    argnames="where",
    argvalues=[
        True,
        False,
    ],
)
@pytest.mark.parametrize("proportion", [0.25, 0.45])
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=[
        "mirror",
        "nearest",
        "wrap",
        "truncate",
    ],
)
def test_trimmed_mean_filter(
    array: np.ndarray,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...],
    where: bool | np.ndarray,
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
    proportion: float,
):
    if axis is None:
        axis_normalized = tuple(range(array.ndim))
    else:
        try:
            axis_normalized = np.core.numeric.normalize_axis_tuple(
                axis, ndim=array.ndim
            )
        except np.AxisError:
            with pytest.raises(np.AxisError):
                ndfilters.trimmed_mean_filter(
                    array=array,
                    size=size,
                    proportion=proportion,
                    axis=axis,
                    where=where,
                )
            return

    if isinstance(size, int):
        size_normalized = (size,) * len(axis_normalized)
    else:
        size_normalized = size

    if len(size_normalized) != len(axis_normalized):
        with pytest.raises(ValueError):
            ndfilters.trimmed_mean_filter(
                array=array,
                size=size,
                proportion=proportion,
                axis=axis,
                where=where,
            )
        return

    result = ndfilters.trimmed_mean_filter(
        array=array,
        size=size,
        proportion=proportion,
        axis=axis,
        where=where,
        mode=mode,
    )

    assert result.shape == array.shape
    assert result.sum() != 0

    if mode == "truncate":
        return

    if not np.all(where):
        return

    size_scipy = [1] * array.ndim
    for i, ax in enumerate(axis_normalized):
        size_scipy[ax] = size_normalized[i]

    expected = scipy.ndimage.generic_filter(
        input=array,
        function=scipy.stats.trim_mean,
        size=size_scipy,
        mode=mode,
        extra_keywords=dict(proportiontocut=proportion),
    )

    assert np.allclose(result, expected)
