# Cell 1
import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    attn_weights = np.asarray(attn_weights, dtype=float)
    T = attn_weights.shape[0]

    if method == 'tril':
        mask = np.tril(np.ones((T, T)))                        # 1 on/below diagonal
        masked = attn_weights * mask                           # element-wise, future -> 0
        result = masked / masked.sum(axis=-1, keepdims=True)   # rows sum to 1 again

    elif method == 'triu':
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)      # True above diagonal (future)
        scores = np.log(attn_weights)                          # recover pre-softmax scores
        scores[mask] = -np.inf                                 # future -> -inf
        scores = scores - scores.max(axis=-1, keepdims=True)   # stable softmax
        e = np.exp(scores)                                     # exp(-inf) = 0
        result = e / e.sum(axis=-1, keepdims=True)

    else:
        raise ValueError("method must be 'tril' or 'triu'")

    return result.tolist()