# Cell 1
import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    token_size = token_embeddings.shape[1]
    trimmed_pos = pos_embedding_matrix[:token_size]
    return token_embeddings + trimmed_pos
    pass