import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    logs = true_labels * np.log(np.clip(predicted_probs,epsilon,1-epsilon))
    sum = -np.sum(logs,axis = 1)
    return np.mean(sum)
    pass