import pytest
import numpy as np
import scipy.ndimage
import scipy.stats
import ndfilters


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.random.random(16),
        np.random.random((16, 17)),
        np.random.random((16, 17, 18)),
    ],
)
@pytest.mark.parametrize(
    argnames="kernel_shape",
    argvalues=[5, (5,), (5, 6), (5, 6, 7)],
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
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
@pytest.mark.parametrize("proportion", [0.25, 0.45])
def test_trimmed_mean_filter(
    array: np.ndarray,
    kernel_shape: int | tuple[int, ...],
    axis: None | int | tuple[int, ...],
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
                    kernel_shape=kernel_shape,
                    proportion=proportion,
                    axis=axis,
                )
            return

    kernel_shape_normalized = (
        (kernel_shape,) * len(axis_normalized)
        if isinstance(kernel_shape, int)
        else kernel_shape
    )

    if len(kernel_shape_normalized) != len(axis_normalized):
        with pytest.raises(ValueError):
            ndfilters.trimmed_mean_filter(
                array=array,
                kernel_shape=kernel_shape,
                proportion=proportion,
                axis=axis,
            )
        return

    result = ndfilters.trimmed_mean_filter(
        array=array,
        kernel_shape=kernel_shape,
        proportion=proportion,
        axis=axis,
    )

    kernel_shape_scipy = [1] * array.ndim
    for i, ax in enumerate(axis_normalized):
        kernel_shape_scipy[ax] = kernel_shape_normalized[i]

    expected = scipy.ndimage.generic_filter(
        input=array,
        function=scipy.stats.trim_mean,
        size=kernel_shape_scipy,
        mode="mirror",
        extra_keywords=dict(proportiontocut=proportion),
    )

    assert np.allclose(result, expected)
