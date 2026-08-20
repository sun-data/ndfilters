import pytest
import numpy as np
import scipy
import astropy.units as u
import ndfilters


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.random.random((5, 6, 7)),
        np.random.random((5, 6, 7)) * u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="kernel,axis",
    argvalues=[
        (np.array([1, 2, 1]), ~0),
        (np.array([1, 2, 3]) / 6, ~0),
        (np.random.random((3,)), ~0),
        (np.random.random((3, 4)), (~1, ~0)),
        (np.random.random((3, 4, 5)), None),
    ],
)
@pytest.mark.parametrize(
    argnames="where",
    argvalues=[
        True,
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
def test_convolve(
    array: np.ndarray | u.Quantity,
    kernel: np.ndarray | u.Quantity,
    axis: None | int | tuple[int, ...],
    where: bool | np.ndarray,
    mode: str,
):

    kwargs = dict(
        array=array,
        kernel=kernel,
        axis=axis,
        where=where,
        mode=mode,
    )

    result = ndfilters.convolve(**kwargs)

    assert result.sum() != 0

    if mode == "truncate":
        return

    axis_ = axis
    if axis_ is None:
        axis_ = np.arange(array.ndim)
    axis_ = np.lib.array_utils.normalize_axis_tuple(axis_, ndim=array.ndim)

    axis_orthogonal = [ax for ax in range(array.ndim) if ax not in axis_]

    kernel_ = np.expand_dims(kernel, axis=axis_orthogonal)

    result_expected = scipy.ndimage.convolve(
        input=array,
        weights=kernel_,
        mode=mode,
    )

    assert np.allclose(u.Quantity(result).value, result_expected)


def test_convolve_mode_invalid():
    with pytest.raises(ValueError, match="Unrecognized mode="):
        ndfilters.convolve(np.random.uniform(size=11), np.ones(3) / 3, mode="foo")


@pytest.mark.parametrize(
    argnames="size_array",
    argvalues=[1, 2, 5],
)
@pytest.mark.parametrize(
    argnames="size_kernel",
    argvalues=[3, 5, 9, 11, 13, 21],
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=["mirror", "nearest", "wrap"],
)
def test_convolve_kernel_wider_than_array(
    size_array: int,
    size_kernel: int,
    mode: str,
):
    """
    A kernel wider than ``2 * size_array - 1`` places an index more than one
    array width outside the boundary, which used to be rectified to an
    out-of-bounds index in "mirror" mode.
    """
    array = np.random.uniform(size=size_array)
    kernel = np.random.uniform(size=size_kernel)

    result = ndfilters.convolve(array, kernel, mode=mode)

    expected = scipy.ndimage.convolve(array, kernel, mode=mode)

    assert np.allclose(result, expected)
