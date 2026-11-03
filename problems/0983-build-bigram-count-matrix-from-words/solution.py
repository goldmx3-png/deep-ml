# Cell 1
import numpy as np

def bigram_counts(words):
    chars = set()
    for word in words:
        for ch in word:
            chars.add(ch)

    vocab = ['.'] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    n = len(vocab)

    ans = np.zeros((n, n), dtype=int)

    for word in words:
        seq = '.' + word + '.'
        for first, second in zip(seq, seq[1:]):
            ans[stoi[first], stoi[second]] += 1

    return ans.tolist()