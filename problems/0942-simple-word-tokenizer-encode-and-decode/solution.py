import re

def encode(text, vocab):
    pattern = '([,.:;?_!"()\']|--|\s)'
    tokens = re.split(pattern,text)
    stripped_tokens = [i.strip() for i in tokens if i]
    # token_map = {}
    # j=0
    ans = []
    # print(stripped_tokens)
    for i in stripped_tokens:
        if i in vocab:
            ans.append(vocab[i])
    return ans
            
    pass

def decode(ids, vocab):
    inverse_vocab = {value: key for key, value in vocab.items()}

    tokens = [inverse_vocab[i] for i in ids]

    text = " ".join(tokens)

    text = re.sub(r'\s+([,.!?:;"\'()])', r'\1', text)

    return text


'''

['hello', 
',', 
'', 
' ', 
'world', 
'!', 
'']

'''