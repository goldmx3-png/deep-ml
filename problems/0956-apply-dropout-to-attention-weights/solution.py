# Cell 1
import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    attn_weights = np.asarray(attn_weights, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask)

    if dropout_rate == 0:
        dropped_weights = attn_weights
    else:
        dropped_weights = attn_weights * mask / (1 - dropout_rate)

    return (dropped_weights @ values).tolist()
    pass