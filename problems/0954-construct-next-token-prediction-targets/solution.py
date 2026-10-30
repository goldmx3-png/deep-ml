# Cell 1
def make_input_target(token_ids, context_size):
    """Build next-token prediction (input, target) pair.

    Args:
        token_ids: list[int] of token IDs (length >= context_size + 1)
        context_size: int, number of tokens in the input window
    Returns:
        list: [x, y] where x and y are lists of length context_size,
              with y being x shifted right by one position.
    """
    n = len(token_ids)
    if n < context_size+1:
        raise ValueError()
    if n == 0:
        return [[],[]]
    ans = []
    for i in range(n):
        if i>1:
            break
        ans.append(token_ids[i:i+context_size])
    return ans
    pass