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


def _mean_masked(
    array: np.ndarray,
    size: int | tuple[int, ...],
    axis: None | int | tuple[int, ...],
    where: np.ndarray,
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
) -> np.ndarray:
    """
    The mean of the elements of each kernel footprint which are not excluded
    by `where`, the ratio of their sum to their number, both calculated with
    :func:`scipy.ndimage.uniform_filter`.
    Its ``"constant"`` mode puts zeros outside the array, which leaves out the
    parts of the kernel outside the array from both, as ``"truncate"`` does.
    """
    if axis is None:
        axis = tuple(range(array.ndim))
    axis = np.lib.array_utils.normalize_axis_tuple(axis, ndim=array.ndim)

    if isinstance(size, int):
        size = (size,) * len(axis)

    size_scipy = [1] * array.ndim
    for i, ax in enumerate(axis):
        size_scipy[ax] = size[i]

    mode_scipy = "constant" if mode == "truncate" else mode
    total = scipy.ndimage.uniform_filter(
        np.where(where, array, 0),
        size=size_scipy,
        mode=mode_scipy,
    )
    num = scipy.ndimage.uniform_filter(
        where.astype(float),
        size=size_scipy,
        mode=mode_scipy,
    )
    return np.divide(
        total,
        num,
        out=np.full(array.shape, np.nan),
        where=num > 0.5 / np.prod(size_scipy),
    )


@pytest.mark.parametrize(
    argnames="shape,axis,size",
    argvalues=[
        ((11,), None, 3),
        ((11,), None, 4),
        ((11, 12), None, (3, 4)),
        ((7, 8, 9), None, 3),
        ((7, 8, 9), None, (4, 3, 4)),
        ((3, 11, 12), (1, 2), (3, 4)),
        ((7, 8, 9), (0, 2), (4, 3)),
        ((7, 8, 9), -1, 4),
    ],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=["mirror", "nearest", "wrap", "truncate"],
)
@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=[np.float32, np.float64],
)
def test_mean_filter_where(
    shape: tuple[int, ...],
    axis: None | int | tuple[int, ...],
    size: int | tuple[int, ...],
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
    dtype: type[np.floating],
) -> None:
    """
    The mean of each kernel footprint leaves out the elements excluded by
    `where`, and the mean of a footprint whose elements are all excluded is
    NaN, including along a subset of the axes and with a different size along
    each.
    """
    rng = np.random.default_rng(0)
    array = rng.uniform(size=shape).astype(dtype)
    where = rng.uniform(size=shape) > 0.3
    where[(slice(0, 5),) * len(shape)] = False

    result = ndfilters.mean_filter(
        array,
        size=size,
        axis=axis,
        where=where,
        mode=mode,
    )

    assert result.dtype == dtype
    assert np.isnan(result[(2,) * len(shape)])
    expected = _mean_masked(array, size=size, axis=axis, where=where, mode=mode)
    assert np.allclose(result, expected, equal_nan=True)
