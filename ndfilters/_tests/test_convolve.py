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
        (np.random.random((2,)), ~0),
        (np.random.random((2, 3)), (~1, ~0)),
        (np.random.random((2, 3, 4)), (0, 1, ~0)),
    ],
)
@pytest.mark.parametrize(
    argnames="where",
    argvalues=[
        True,
    ]
)
@pytest.mark.parametrize(
    argnames="mode",
    argvalues=[
        "mirror",
        "nearest",
        "wrap",
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

    axis_ = axis
    if axis_ is None:
        axis_ = np.arange(array.ndim)
    axis_ = np.core.numeric.normalize_axis_tuple(axis_, ndim=array.ndim)

    axis_orthogonal = [ax for ax in range(array.ndim) if ax not in axis_]

    kernel_ = np.expand_dims(kernel, axis=axis_orthogonal)

    result_expected = scipy.ndimage.convolve(
        input=array,
        weights=kernel_,
        mode=mode,
    )

    assert np.all(u.Quantity(result).value == result_expected)
