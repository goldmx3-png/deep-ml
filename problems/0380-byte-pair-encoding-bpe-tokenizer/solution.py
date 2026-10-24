from collections import Counter

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    # Convert each word into a list of tokens
    words = {
        tuple(word.split()): freq
        for word, freq in corpus.items()
    }

    merges = []

    for _ in range(num_merges):
        pair_counts = Counter()

        # Step 1: Count adjacent pairs
        for tokens, freq in words.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        # No pairs left to merge
        if not pair_counts:
            break

        # Step 2: Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)

        merges.append(best_pair)

        # Step 3: Merge in all words
        new_words = {}

        for tokens, freq in words.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            key = tuple(new_tokens)
            new_words[key] = new_words.get(key, 0) + freq

        words = new_words

    return merges