# Cell 1
import numpy as np
from scipy.special import softmax

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    d_k = K.shape[1]
    R = Q @ K.T
    logits = R/np.sqrt(d_k)
    return np.round(softmax(logits,axis=-1),4).tolist()
    pass