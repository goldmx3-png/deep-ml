# Cell 1
import numpy as np
from scipy.special import softmax

def simple_self_attention(X):
    Y = np.array(X, dtype=float)
    scores = Y @ Y.T
    weights = softmax(scores, axis=1)
    return (weights @ Y).tolist()