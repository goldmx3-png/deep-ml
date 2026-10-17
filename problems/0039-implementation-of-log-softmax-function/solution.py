import numpy as np
from scipy.special import log_softmax as scipy_log_softmax

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	return scipy_log_softmax(scores)
	pass