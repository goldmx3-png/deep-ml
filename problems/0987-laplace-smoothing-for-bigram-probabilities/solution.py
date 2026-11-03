# Cell 1
import numpy as np

def smooth_bigram_probs(N, k):
    result = np.array(N, dtype=float) + k
    ans = np.zeros_like(result)
    for i in range(len(result)):
        row_sum = result[i].sum()
        for j in range(len(result[i])):
            ans[i][j] = result[i][j] / row_sum
    return ans.tolist()