# Cell 1
import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    V = vocab_size
    N = len(xs)

    # The model: one score per (current char, next char) pair, starting at "know nothing"
    W = np.zeros((V, V))

    # One-hot encoding: row i of the identity matrix has its 1 in column i,
    # so picking rows xs gives "row i has a 1 in column xs[i]"
    X = np.eye(V)[xs]          # (N, V) what each example saw
    Y = np.eye(V)[ys]          # (N, V) the right answer for each example

    for _ in range(num_iters):
        # Forward: one-hot @ W picks out each example's row of scores
        Z = X @ W                                   # (N, V) logits

        # Softmax row by row (subtract the row max first for numerical safety)
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        P = E / E.sum(axis=1, keepdims=True)        # (N, V) probabilities

        # Gradient: error per example, routed by X.T to the row it came from,
        # averaged over N, plus the L2 pull toward zero
        grad = X.T @ (P - Y) / N + (2 * alpha / V**2) * W   # (V, V)

        # Step downhill
        W = W - lr * grad

    return W.tolist()