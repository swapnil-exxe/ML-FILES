import pytest
import numpy as np
from utils.dataset_analyzer import calculate_skewness, audit_dataset_health

def test_skewness_calculation():
    symmetric = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(calculate_skewness(symmetric), 0.0, atol=1e-5)

def test_dataset_health_audit():
    mat = np.array([
        [1.0, 10.0],
        [2.0, np.nan],
        [3.0, 30.0],
        [4.0, 40.0]
    ])

    report = audit_dataset_health(mat)
    assert report["n_rows"] == 4
    assert report["n_cols"] == 2
    assert report["total_nans"] == 1
    assert report["nan_percentage"] == 12.5
