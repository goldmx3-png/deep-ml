# Cell 1
import numpy as np
from typing import Tuple

def compute_qkv(X, W_q, W_k, W_v):
    X = np.asarray(X, dtype=float)
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V

def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[-1]
    head_dim = d_model // n_heads
    outs = []
    for h in range(n_heads):
        s = slice(h * head_dim, (h + 1) * head_dim)
        outs.append(self_attention(Q[:, s], K[:, s], V[:, s]))
    return np.concatenate(outs, axis=-1)