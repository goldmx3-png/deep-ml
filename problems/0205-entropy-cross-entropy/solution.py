import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution
        Q: Predicted probability distribution
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    P_arr = np.array(P)
    Q_arr = np.array(Q)
    
    # 1. Calculate Entropy H(P) safely ignoring 0s
    logs = np.log(P_arr, out=np.zeros_like(P_arr), where=P_arr > 0)
    entropy = - np.sum(P_arr * logs)
    
    # 2. Calculate Cross-Entropy H(P, Q) safely ignoring 0s
    Q_logs = np.log(Q_arr, out=np.zeros_like(Q_arr), where=Q_arr > 0)
    cross_entropy = - np.sum(P_arr * Q_logs)
    
    return (float(entropy), float(cross_entropy))
