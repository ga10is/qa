import code
import unicodedata
import functools

import numpy as np
import scipy

from ..unqg.sp import SentencePieceTokenizer
from . import core
from .indexer import count
from .documents import SimpleDocuments, Documents


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

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_idxs = res.indices[o_sort]
        return doc_idxs, doc_scores


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


def print_retrieve_result(docs: Documents, doc_indices, doc_scores):
    """
    Parameters
    ----------
    docs: dict
        {doc_id: doc_text}
    """
    doc_texts = [docs.get_text_by_idx(idx) for idx in doc_indices]
    for doc_text in doc_texts:
        print(doc_text)


def main():
    # load documents
    documents = SimpleDocuments.load('data/qas/documents.json')

    # initialize retriever
    core.Pickle.load('data/qas/index.pkl')
    normalizer = functools.partial(func=unicodedata.normalize, __form='NFKC')
    tokenizer = SentencePieceTokenizer('data/qas/spm.3000.model', 'id')
    r = Retriever(normalizer, tokenizer)

    # retrieve
    query = ''
    doc_indices, doc_scores = r.closest_docs(query)

    # print
    print_retrieve_result(documents, doc_indices, doc_scores)


if __name__ == '__main__':
    banner = 'retrive document'
    code.interact(banner, local=locals())
    main()
