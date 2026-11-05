# Cell 1
import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
    N = X.shape[0]                    # Number of contexts

    embeddings = C[X]                 # (N, block_size, emb_dim)
    flat = embeddings.reshape(N, -1)  # (N, block_size * emb_dim)

    hidden = np.tanh(flat @ W1 + b1)  # (N, hidden_size)
    logits = hidden @ W2 + b2         # (N, vocab_size)

    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True)
    )

    correct_log_probs = log_probs[np.arange(N), Y]
    return float(-correct_log_probs.mean())