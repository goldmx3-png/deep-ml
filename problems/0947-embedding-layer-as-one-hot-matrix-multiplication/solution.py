# Cell 1
import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    grid = np.zeros((len(token_ids),W.shape[0]))
    # print(grid)
    i=0
    for token in token_ids:
        grid[i][token]=1
        i+=1
    # print(grid)
    
    return grid @ W
    pass