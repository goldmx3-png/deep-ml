# Cell 1
import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    arr = np.asarray(arr, dtype=float)

    if norm_type == 'l1':
        return float(np.sum(np.abs(arr)))
    if norm_type == 'l2':
        return float(np.sqrt(np.sum(arr ** 2)))
    if norm_type == 'linf':
        return float(np.max(np.abs(arr)))
    if norm_type == 'frobenius':
        if arr.ndim != 2:
            raise ValueError("Frobenius norm requires a 2D array")
        return float(np.sqrt(np.sum(arr ** 2)))

    raise ValueError(f"Unknown norm_type: {norm_type}")