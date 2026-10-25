# --- cell 1 ---
import math
from collections import Counter


def train_tokenizer(corpus, vocab_size):
    """
    BPE tokenizer whose merge count is tuned for a bigram language model.

    1. Base vocab = every character seen in the corpus, plus one ESC token.
       Characters never seen in training are encoded losslessly as
       ESC <codepoint digits as base-tokens> ESC, so decode(encode(s)) == s
       for any string.
    2. BPE merges are learned up to the vocab budget.
    3. More merges is not always better for a bigram model: long tokens get
       sparse. So we hold out 10% of the corpus, train BPE on the rest, and
       measure smoothed bigram BPC for many merge counts. The best count is
       then applied to merges learned on the full corpus.
    """
    corpus = [d for d in corpus if isinstance(d, str)]
    chars = sorted(set("".join(corpus)))
    if not chars:
        chars = ["a"]

    n_base = len(chars)
    ESC = n_base                        # escape token for unseen characters
    max_merges = max(0, vocab_size - n_base - 1)
    if n_base + 1 > vocab_size:
        raise ValueError("vocab_size too small for the corpus alphabet")

    # ---------------------------------------------------------------- BPE
    def learn_merges(docs, n_merges):
        """Return ordered list of merges: (left_id, right_id, new_id)."""
        cid = {c: i for i, c in enumerate(chars)}
        words = Counter(docs)
        seqs = [[cid[c] for c in w] for w in words]
        freqs = list(words.values())
        merges = []
        next_id = n_base + 1
        for _ in range(n_merges):
            pairs = Counter()
            for s, f in zip(seqs, freqs):
                for i in range(len(s) - 1):
                    pairs[(s[i], s[i + 1])] += f
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new = next_id
            next_id += 1
            merges.append((a, b, new))
            for k, s in enumerate(seqs):
                if len(s) < 2:
                    continue
                out, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[k] = out
        return merges

    def make_codec(merges):
        cid = {c: i for i, c in enumerate(chars)}
        rank = {(a, b): r for r, (a, b, _) in enumerate(merges)}
        result = {(a, b): new for a, b, new in merges}
        strings = {i: c for i, c in enumerate(chars)}
        for a, b, new in merges:
            strings[new] = strings[a] + strings[b]
        base = n_base  # digit base for escaped code points
        cache = {}

        def bpe(ids):
            while len(ids) > 1:
                best, bi = None, -1
                for i in range(len(ids) - 1):
                    r = rank.get((ids[i], ids[i + 1]))
                    if r is not None and (best is None or r < best):
                        best, bi = r, i
                if best is None:
                    break
                pair = (ids[bi], ids[bi + 1])
                new = result[pair]
                out, i = [], 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                        out.append(new)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                ids = out
            return ids

        def encode_chunk(chunk):
            if chunk not in cache:
                cache[chunk] = bpe([cid[c] for c in chunk])
            return cache[chunk]

        def encode(text):
            out, run = [], []
            for c in text:
                if c in cid:
                    run.append(c)
                    continue
                if run:
                    out.extend(encode_chunk("".join(run)))
                    run = []
                # escape unseen char: ESC digits... ESC (digits are base ids)
                cp, digits = ord(c), []
                while True:
                    digits.append(cp % base)
                    cp //= base
                    if cp == 0:
                        break
                out.append(ESC)
                out.extend(reversed(digits))
                out.append(ESC)
            if run:
                out.extend(encode_chunk("".join(run)))
            return out

        def decode(ids):
            out, i, n = [], 0, len(ids)
            while i < n:
                t = ids[i]
                if t == ESC:
                    j, cp = i + 1, 0
                    while j < n and ids[j] != ESC:
                        cp = cp * base + ids[j]
                        j += 1
                    out.append(chr(cp))
                    i = j + 1
                else:
                    out.append(strings[t])
                    i += 1
            return "".join(out)

        return encode, decode

    # ------------------------------------------- tune merge count on held-out
    def bigram_bpc(encode, train_docs, val_docs, V, alpha=0.1):
        BOS = V  # document boundary symbol
        uni, bi = Counter(), Counter()
        for d in train_docs:
            seq = [BOS] + encode(d) + [BOS]
            for x, y in zip(seq, seq[1:]):
                bi[(x, y)] += 1
                uni[x] += 1
        K = V + 1
        bits, nchar = 0.0, 0
        for d in val_docs:
            seq = [BOS] + encode(d) + [BOS]
            for x, y in zip(seq, seq[1:]):
                p = (bi[(x, y)] + alpha) / (uni[x] + alpha * K)
                bits -= math.log2(p)
            nchar += len(d) + 1  # +1 for the end-of-document
        return bits / max(nchar, 1)

    best_k = max_merges
    if max_merges > 0 and len(corpus) >= 50:
        split = int(len(corpus) * 0.9)
        # deterministic interleaved split so the held-out set is representative
        tr = [d for i, d in enumerate(corpus) if i % 10 != 0]
        va = [d for i, d in enumerate(corpus) if i % 10 == 0]
        tr_merges = learn_merges(tr, max_merges)
        step = max(1, len(tr_merges) // 20)
        candidates = sorted(set(list(range(0, len(tr_merges) + 1, step))
                                + [len(tr_merges)]))
        best_score = float("inf")
        for k in candidates:
            enc, _ = make_codec(tr_merges[:k])
            score = bigram_bpc(enc, tr, va, n_base + 1 + k)
            if score < best_score:
                best_score, best_k = score, k

    merges = learn_merges(corpus, best_k)
    return make_codec(merges)