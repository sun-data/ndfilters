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
            axis_normalized = np.lib.array_utils.normalize_axis_tuple(
                axis, ndim=array.ndim
            )
        except np.exceptions.AxisError:
            with pytest.raises(np.exceptions.AxisError):
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


def _trim_mean_nan_last(a: np.ndarray, proportiontocut: float) -> float:
    """
    The trimmed mean of `a`, with NaN sorting after every number, as in
    :func:`numpy.sort`, so that it is trimmed with the largest elements.
    :func:`scipy.stats.trim_mean` instead returns NaN if `a` holds any.
    """
    a = np.sort(a)
    lowercut = int(proportiontocut * a.size)
    return float(np.mean(a[lowercut : a.size - lowercut]))


@pytest.mark.parametrize("size", [3, 4, (3, 5)])
@pytest.mark.parametrize("proportion", [0.1, 0.25, 0.45])
@pytest.mark.parametrize("mode", ["mirror", "nearest", "wrap"])
def test_trimmed_mean_filter_nan(
    size: int | tuple[int, ...],
    proportion: float,
    mode: Literal["mirror", "nearest", "wrap"],
) -> None:
    rng = np.random.default_rng(0)
    array = rng.normal(size=(9, 10))
    array[rng.random(array.shape) < 0.15] = np.nan
    array[rng.random(array.shape) < 0.05] = np.inf

    result = ndfilters.trimmed_mean_filter(
        array=array,
        size=size,
        mode=mode,
        proportion=proportion,
    )

    expected = scipy.ndimage.generic_filter(
        input=array,
        function=_trim_mean_nan_last,
        size=size,
        mode=mode,
        extra_keywords=dict(proportiontocut=proportion),
    )

    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("proportion", [-0.1, 0.6, np.nan])
def test_trimmed_mean_filter_proportion_invalid(proportion: float) -> None:
    """
    A proportion outside 0 to 0.5 raises :class:`ValueError` before filtering,
    since an exception raised in the parallel loops would not reach the caller.
    """
    with pytest.raises(ValueError, match="proportion"):
        ndfilters.trimmed_mean_filter(
            array=np.random.random(10),
            size=3,
            proportion=proportion,
        )


def test_trimmed_mean_filter_proportion_half() -> None:
    """
    Cutting half of an odd-sized footprint from each end leaves its median.
    """
    array = np.random.default_rng(0).random((9, 10))

    result = ndfilters.trimmed_mean_filter(array, size=(3, 5), proportion=0.5)

    assert np.array_equal(result, ndfilters.median_filter(array, size=(3, 5)))


def test_trimmed_mean_filter_array_unchanged() -> None:
    """
    The trimmed mean of each footprint reorders the copy of it which
    :func:`ndfilters.generic_filter` makes, not `array` itself.
    """
    array = np.random.default_rng(0).random(11)
    array_original = array.copy()

    ndfilters.trimmed_mean_filter(array, size=5, proportion=0.25)

    assert np.array_equal(array, array_original)
