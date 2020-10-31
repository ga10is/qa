from sklearn.utils import murmurhash3_32

from typing import List


def get_ngrams(n, tokens: List[str]):
    ngram_idx = [(s, e + 1)
                 for s in range(len(tokens))
                 for e in range(s, min(s + n, len(tokens)))]
    ngrams = ['_'.join(tokens[s: e]) for (s, e) in ngram_idx]
    return ngrams


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets
