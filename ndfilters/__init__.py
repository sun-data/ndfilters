"""
Numba-accelerated, n-dimensional filters similar to those in :mod:`scipy.ndimage`.
"""

from ._generic import generic_filter
from ._mean import mean_filter
from ._trimmed_mean import trimmed_mean_filter
from ._variance import variance_filter

__all__ = [
    "generic_filter",
    "mean_filter",
    "trimmed_mean_filter",
    "variance_filter",
]
