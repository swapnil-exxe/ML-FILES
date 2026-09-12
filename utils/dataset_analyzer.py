import numpy as np

def calculate_skewness(data: np.ndarray) -> float:
    """Calculates Fisher-Pearson coefficient of skewness for a 1D numeric array."""
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 3:
        return 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0

    skew = (np.sum((data - mean) ** 3) / n) / (std ** 3)
    return float(skew)

def audit_dataset_health(matrix: np.ndarray) -> dict:
    """
    Audits 2D dataset matrix feature health including total rows, total columns, 
    NaN percentage, and feature skewness.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    n_rows, n_cols = matrix.shape
    total_elements = matrix.size
    nan_count = int(np.sum(np.isnan(matrix)))
    nan_percentage = float((nan_count / total_elements) * 100.0) if total_elements > 0 else 0.0

    skews = [calculate_skewness(matrix[:, col]) for col in range(n_cols)]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_nans": nan_count,
        "nan_percentage": nan_percentage,
        "feature_skewness": skews
    }
