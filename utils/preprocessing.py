import numpy as np

def min_max_scale(data: np.ndarray, feature_range=(0, 1)) -> np.ndarray:
    """Scales numeric array features linearly into feature_range [min, max]."""
    data = np.asarray(data, dtype=np.float64)
    min_val, max_val = np.min(data), np.max(data)
    if np.isclose(min_val, max_val):
        return np.zeros_like(data)
    
    scaled = (data - min_val) / (max_val - min_val)
    target_min, target_max = feature_range
    return scaled * (target_max - target_min) + target_min

def handle_missing_values(data: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Imputes np.nan values in numeric vector using mean or median strategy."""
    data = np.asarray(data, dtype=np.float64).copy()
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return data

    valid_vals = data[~nan_mask]
    if len(valid_vals) == 0:
        return np.zeros_like(data)

    fill_val = np.mean(valid_vals) if strategy == "mean" else np.median(valid_vals)
    data[nan_mask] = fill_val
    return data
