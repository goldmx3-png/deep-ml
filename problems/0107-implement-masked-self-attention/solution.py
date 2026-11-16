# Cell 1
import numpy as np
from scipy.special import softmax

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k) + mask
    probs = softmax(scores, axis=1)
    return probs @ V