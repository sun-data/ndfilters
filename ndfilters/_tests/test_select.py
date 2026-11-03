import pytest
import numpy as np
from ndfilters._select import sorts_before, select

_rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    argnames="a,b,expected",
    argvalues=[
        (1.0, 2.0, True),
        (2.0, 1.0, False),
        (1.0, 1.0, False),
        (-np.inf, 0.0, True),
        (np.inf, np.nan, True),
        (np.nan, np.inf, False),
        (np.nan, np.nan, False),
    ],
)
def test_sorts_before(a: float, b: float, expected: bool) -> None:
    assert sorts_before(a, b) == expected


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        np.array([1.0]),
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
        _rng.normal(size=25),
        np.round(_rng.normal(size=30)),
        np.arange(20.0),
        np.arange(20.0)[::-1].copy(),
        np.zeros(17),
        np.array([np.nan, 1.0, np.inf, -np.inf, np.nan, 0.0, -1.0, 1.0]),
        np.full(5, np.nan),
    ],
)
def test_select(array: np.ndarray) -> None:
    expected = np.sort(array)
    for k in range(array.size):
        a = array.copy()
        select(a, k, 0, a.size - 1)
        assert np.array_equal(np.sort(a), expected, equal_nan=True)
        assert np.array_equal(a[k], expected[k], equal_nan=True)
        assert not any(sorts_before(a[k], x) for x in a[:k])
        assert not any(sorts_before(x, a[k]) for x in a[k + 1 :])


@pytest.mark.parametrize(
    argnames="low,high",
    argvalues=[
        (0, 9),
        (3, 9),
        (0, 5),
        (2, 7),
        (4, 4),
    ],
)
def test_select_interval(low: int, high: int) -> None:
    array = _rng.normal(size=10)
    expected = np.sort(array[low : high + 1])
    for k in range(low, high + 1):
        a = array.copy()
        select(a, k, low, high)
        assert np.array_equal(a[:low], array[:low])
        assert np.array_equal(a[high + 1 :], array[high + 1 :])
        assert np.array_equal(np.sort(a[low : high + 1]), expected)
        assert a[k] == expected[k - low]
        assert np.all(a[low:k] <= a[k])
        assert np.all(a[k + 1 : high + 1] >= a[k])
