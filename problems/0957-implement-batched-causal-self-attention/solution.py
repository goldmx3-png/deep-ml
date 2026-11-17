# Cell 1
import numpy as np
from scipy.special import softmax

def batched_causal_self_attention(X, W_query, W_key, W_value):
    X = np.array(X)
    Wq, Wk, Wv = np.array(W_query), np.array(W_key), np.array(W_value)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    dout = Wk.shape[1]

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dout)   # (B, T, T)

    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T)) * (-np.inf), k=1)     # (T, T)
    scores = scores + mask                               # broadcasts over B

    probs = softmax(scores, axis=-1)
    return (probs @ V).tolist()