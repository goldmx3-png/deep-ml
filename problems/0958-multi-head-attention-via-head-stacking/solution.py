# Cell 1
import numpy as np

def multi_head_attention_stack(x, heads):
    x = np.array(x)
    T = x.shape[1]
    mask = np.triu(np.full((T, T), -np.inf), k=1)

    ans = []
    for Wq, Wk, Wv in heads:
        Wq, Wk, Wv = np.array(Wq), np.array(Wk), np.array(Wv)
        Q, K, V = x @ Wq, x @ Wk, x @ Wv
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(K.shape[-1]) + mask
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        ans.append(probs @ V)

    return np.concatenate(ans, axis=-1)