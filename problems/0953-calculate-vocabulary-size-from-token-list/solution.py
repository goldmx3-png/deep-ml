# from collections import Counter

def vocab_size(tokens, special_tokens=None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    if special_tokens and len(special_tokens)>0:
        tokens.extend(special_tokens)
    counts = set(tokens)
    return len(counts)
    pass