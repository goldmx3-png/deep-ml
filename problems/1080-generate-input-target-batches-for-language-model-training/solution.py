# Cell 1
import numpy as np

def get_batch(data: np.ndarray, block_size: int,
              batch_size: int, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)

    starts = rng.integers(
        0,
        len(data) - block_size,
        size=batch_size
    )

    input_rows = []
    target_rows = []

    for i in starts:
        input_rows.append(data[i:i + block_size])
        target_rows.append(data[i + 1:i + 1 + block_size])

    x = np.stack(input_rows)
    y = np.stack(target_rows)

    return np.stack([x, y])