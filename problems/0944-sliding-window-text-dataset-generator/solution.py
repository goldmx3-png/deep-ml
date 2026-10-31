# Cell 1
# Cell 1
def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    n = len(token_ids)
    ans = []
    i=0
    while i<n:
        if i>=n:
            break
        if i+max_length+1<=n:
            input = token_ids[i:i+max_length]
            target = token_ids[i+1:i+max_length+1]
            ans.append((input,target))
        i = i+stride
    return ans
    pass