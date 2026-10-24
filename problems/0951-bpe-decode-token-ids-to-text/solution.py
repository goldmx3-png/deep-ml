def bpe_decode(ids, vocab):
    tokens = []
    for id in ids:
        if id in vocab:
            word = vocab[id]
            # If it starts with 'Ġ', replace it. If not, it safely changes nothing.
            if word.startswith('G'):
                word = word.replace('G', ' ', 1)
            tokens.append(word)
            
    return "".join(tokens)
