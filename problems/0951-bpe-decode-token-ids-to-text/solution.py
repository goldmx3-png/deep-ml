def bpe_decode(ids, vocab):
    text = ""
    for id in ids:
        if id in vocab:
            word = vocab[id]
            # Fix 1: You must assign the result back to a variable!
            if word[0] == 'G':  # Note: BPE usually uses 'Ġ' instead of 'G'
                word = word.replace("G", " ", 1)
            
            text = text + word
    return text