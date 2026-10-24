import re

def encode(text, vocab):
    pattern = r"""([,.:;?_!"()']|--|\s)"""
    tokens = re.split(pattern, text)

    stripped_tokens = [i.strip() for i in tokens if i.strip()]

    ans = []
    for token in stripped_tokens:
        ans.append(vocab.get(token, vocab['<|unk|>']))

    return ans


def decode(ids, vocab):
    inverse_vocab = {value: key for key, value in vocab.items()}

    tokens = [inverse_vocab[i] for i in ids]
    text = " ".join(tokens)

    text = re.sub(r""" ?([,.?!"()':;])""", r'\1', text)

    return text


def tokenize(vocab, text, mode):
    if mode == 'decode':
        return decode(text, vocab)
    else:
        return encode(text, vocab)