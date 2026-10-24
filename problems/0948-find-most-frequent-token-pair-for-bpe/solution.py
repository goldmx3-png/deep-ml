from collections import Counter

def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    touple_list = []
    for seq in sequences:
        for i in range(len(seq)):
            if i+1<len(seq):
                touple_list.append((seq[i],seq[i+1]))
    # print(touple_list)
    if len(touple_list) == 0:
        return None
    counts = dict(Counter(touple_list))
    return max(counts,key = counts.get)
    pass