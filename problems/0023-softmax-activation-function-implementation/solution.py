import math
import numpy as np
from scipy.special import softmax as scipy_softmax

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    probabilities = scipy_softmax(scores)
    #print(sum(probabilities))
    return probabilities
    pass