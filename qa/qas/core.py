import pickle
import hashlib
from sklearn.utils import murmurhash3_32

from typing import List


def get_ngrams(n, tokens: List[str]):
    ngram_idx = [(s, e + 1)
                 for s in range(len(tokens))
                 for e in range(s, min(s + n, len(tokens)))]
    ngrams = ['_'.join(tokens[s: e]) for (s, e) in ngram_idx]
    return ngrams


def hash_murmurhash(text, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(text, positive=True) % num_buckets


def hash_sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


class Pickle:
    @staticmethod
    def pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


def count_lines(path, blocksize=65536):
    def blocks(f):
        while True:
            b = f.read(blocksize)
            if b:
                yield b
            else:
                break

    with open(path, 'r') as f:
        return sum(bl.count('\n') for bl in blocks(f))
