# Cell 1
import numpy as np

def token_embedding_lookup(
    vocab_size: int,
    embed_dim: int,
    token_ids: list,
    seed: int = 0
) -> list:
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((vocab_size, embed_dim))
    return weights[token_ids].tolist()