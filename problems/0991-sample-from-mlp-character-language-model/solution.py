# Cell 1
import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    C = np.asarray(C)
    context = [0] * block_size
    rng = np.random.default_rng(seed)
    output = []

    for _ in range(max_tokens):
        e = C[context].reshape(-1)
        h = np.tanh(e @ W1 + b1)
        logits = h @ W2 + b2

        weights = np.exp(logits - np.max(logits))
        probs = weights / weights.sum()
        ix = rng.choice(len(probs), p=probs)

        if ix == 0:
            break

        output.append(itos[ix])
        context = context[1:] + [ix]

    return "".join(output)