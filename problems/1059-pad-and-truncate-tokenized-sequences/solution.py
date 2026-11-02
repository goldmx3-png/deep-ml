# Cell 1
def pad_and_truncate(sequences, pad_token_id, max_length=None):
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """
    ans = []
    sizes = []
    if max_length is None:
        sizes = [len(seq) for seq in sequences]
        if len(sizes):
            max_length = max(sizes)
    for seq in sequences:
        processed_seq = process_seq(seq, pad_token_id, max_length)
        ans.append(processed_seq)
    return ans
    pass
def process_seq(seq, pad_token_id, max_length):
    if max_length == 0:
        return []
    n = len(seq)
    if n > max_length:
        return seq[:max_length]
    return seq + [pad_token_id] * (max_length - n)