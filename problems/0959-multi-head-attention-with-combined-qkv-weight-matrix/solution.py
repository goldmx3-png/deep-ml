# Cell 1
import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    x, W_qkv, W_out = np.array(x, float), np.array(W_qkv, float), np.array(W_out, float)
    B, N, _ = x.shape
    d_out = W_qkv.shape[1] // 3
    hd = d_out // num_heads

    qkv = (x @ W_qkv).reshape(B, N, 3, num_heads, hd).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                      # (B, H, N, hd)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)     # (B, H, N, N)
    scores = scores + np.triu(np.full((N, N), -np.inf), k=1)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w = e / e.sum(axis=-1, keepdims=True)

    ctx = (w @ v).transpose(0, 2, 1, 3).reshape(B, N, d_out)
    return (ctx @ W_out).tolist()