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
    :func:`scipy.ndimage.correlate`.
    Unlike the running sums of :func:`scipy.ndimage.uniform_filter`, it adds
    the elements of each footprint anew, so NaN and infinities reach only the
    footprints which hold them.
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
    weights = np.ones(size_scipy)
    total = scipy.ndimage.correlate(
        np.where(where, array, 0).astype(float),
        weights=weights,
        mode=mode_scipy,
    )
    num = scipy.ndimage.correlate(
        where.astype(float),
        weights=weights,
        mode=mode_scipy,
    )
    return np.divide(
        total,
        num,
        out=np.full(array.shape, np.nan),
        where=num > 0,
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
        ((5, 5000), None, (3, 4)),
        ((5, 20, 1100), None, (3, 4, 5)),
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
    NaN, including along a subset of the axes, with a different size along
    each, and on arrays long enough that the kernels split them into tiles.
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


# With `NUMBA_DISABLE_JIT=1`, the kernels add numpy scalars, which warn when
# they add infinities of opposite signs.
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar add")
@pytest.mark.parametrize(
    argnames="shape,axis,size",
    argvalues=[
        ((11, 12), None, (3, 4)),
        ((7, 8, 9), None, (4, 3, 4)),
        ((5, 6), None, (3, 13)),
        ((4, 5, 6), None, (9, 3, 14)),
        ((2, 3, 5000), (1, 2), (3, 4)),
        ((2, 2, 9, 1100), (1, 2, 3), (3, 4, 5)),
        ((1, 2, 1100), None, (1, 3, 1030)),
    ],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=["mirror", "nearest", "wrap", "truncate"],
)
def test_mean_filter_nonfinite(
    shape: tuple[int, ...],
    axis: None | int | tuple[int, ...],
    size: int | tuple[int, ...],
    mode: Literal["mirror", "nearest", "wrap", "truncate"],
) -> None:
    """
    The mean of a footprint which holds NaN, or both infinities, is NaN, and
    that of a footprint which holds only one of the infinities is that
    infinity, unless `where` excludes them, including across the boundaries of
    the tiles which the kernels split an array into, for batches of more than
    one element, and for kernels wider than the array or, in 3D, than a tile.
    """
    rng = np.random.default_rng(0)
    array = rng.uniform(size=shape)
    for value in (np.nan, np.inf, -np.inf):
        array[rng.uniform(size=shape) < 0.02] = value
    where = rng.uniform(size=shape) > 0.3

    result = ndfilters.mean_filter(
        array,
        size=size,
        axis=axis,
        where=where,
        mode=mode,
    )

    expected = _mean_masked(array, size=size, axis=axis, where=where, mode=mode)
    assert np.allclose(result, expected, equal_nan=True)
