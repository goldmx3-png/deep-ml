def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """

    unique_tokens = sorted(set(tokens))
    # print("hello->",unique_tokens)
    token_map = {}
    j=0
    for i in unique_tokens:
        token_map.update({i:j})
        j = j+1
    # print(token_map)
    return token_map
    pass
