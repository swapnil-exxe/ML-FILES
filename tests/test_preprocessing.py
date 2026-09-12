import pytest
import numpy as np
from utils.preprocessing import min_max_scale, handle_missing_values

def test_min_max_scale():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled = min_max_scale(arr, feature_range=(0, 1))
    assert np.isclose(np.min(scaled), 0.0)
    assert np.isclose(np.max(scaled), 1.0)
    assert np.isclose(scaled[2], 0.5)

def test_handle_missing_values():
    arr_with_nans = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    imputed = handle_missing_values(arr_with_nans, strategy="mean")
    assert not np.any(np.isnan(imputed))
    # Mean of [1, 2, 4, 5] is 3.0
    assert np.isclose(imputed[2], 3.0)
