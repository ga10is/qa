import unicodedata
import functools

import numpy as np
import scipy

from ..unqg.sp import SentencePieceTokenizer
from .indexer import count


class Retriever:
    def __init__(self, normalizer, tokenizer, index_dict):
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.ngram = index_dict['ngram']
        self.tfidf_matrix = index_dict['tfidf_matrix']

    def to_vec(self, query):
        query = self.normalizer.normalize(query)
        tokens = self.tokenizer.tokenize(query)

        n_words = self.tfidf_matrix.shape[0]
        n_docs = self.tfidf_matrix.shape[1]
        count_vec = get_count_vec(tokens, self.ngram, n_words)
        vec = get_tfidf_vec(count_vec, self.doc_freqs, n_docs)

        return vec

    def closest_docs(self, query, k=1):
        query_vec = self.to_vec(query)
        res = query_vec * self.tfidf_matrix


def get_count_vec(tokens, ngram, hash_size):
    row, col, data = count(
        tokens, doc_idx=0, ngram=ngram, hash_size=hash_size)
    count_vec = scipy.sparse.csr_matrix(
        (data, (row, col)), shape=(1, hash_size)
    )
    return count_vec


def get_tfidf_vec(count_vec, doc_freqs, num_docs):
    """
    Parameters
    ----------
    count_vec: scipy.sparse.csr_matrix, shape of (1, # of word)
    doc_freqs: numpy.ndarray, shape of (# of word,)
    num_docs: int

    Returns
    -------
    scipy.sparse.csr_matrix, shape of (1, # of word)
    """
    # TF
    tfs = np.log1p(count_vec.data)

    # IDF
    print(count_vec.indices)
    Ns = doc_freqs[count_vec.indices]
    idfs = np.log((num_docs - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0

    # TF-IDF
    data = np.multiply(tfs, idfs)

    vec = scipy.sparse.csr_matrix(
        (data, (count_vec.indices, count_vec.indptr)), shape=count_vec.shape
    )
    return vec


def main():
    normalizer = functools.partial(func=unicodedata.normalize, __form='NFKC')
    tokenizer = SentencePieceTokenizer('', 'id')
    r = Retriever(normalizer, tokenizer)
    r.to_vec(query)


if __name__ == '__main__':
    main()
