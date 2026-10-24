import sys


def bpe_encode(text, token_to_id, merges):
    # Your code here
    ids = []
    for ch in text:
        ids.append(token_to_id[ch])
    while True:
        new_ids = []
        i = 0
        flag = False
        while i<len(ids):
            if i+1<len(ids) and (ids[i],ids[i+1]) in merges:
                new_ids.append(merges[(ids[i],ids[i+1])])
                i=i+2
                flag = True
            else:
                new_ids.append(ids[i])
                i=i+1
        if flag:
            ids = new_ids
        else:
            break
    return new_ids
    pass