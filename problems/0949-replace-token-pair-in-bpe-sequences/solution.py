def replace_pair(sequences, pair, new_id):
    ans = []

    for seq in sequences:
        seq_tokens = []
        i = 0

        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                seq_tokens.append(new_id)
                i += 2
            else:
                seq_tokens.append(seq[i])
                i += 1

        ans.append(seq_tokens)

    return ans