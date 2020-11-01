from multiprocessing import Pool
from collections import Counter

import numpy as np
import scipy
from tqdm import tqdm

from . import core
from .documents import SimpleDocuments
from ..unqg.sp import SentencePieceTokenizer

from typing import List


class Indexer:

    def __init__(self, ngram=2):
        self.ngram = ngram

    def doc_to_tfidf_matrix(self, docs: List[List[str]], n_process=1):

        count_matrix = get_count_matrix(docs,
                                        ngram=self.ngram,
                                        hash_size=2 ** 24,
                                        n_process=n_process)
        tfidf_matrix = get_tfidf_matrix(count_matrix)
        doc_freqs = get_doc_freqs(count_matrix)

        self.tfidf_matrix = tfidf_matrix
        self.doc_freqs = doc_freqs

        return tfidf_matrix

    def to_dict(self):
        d = {
            'ngram': self.ngram,
            'doc_freqs': self.doc_freqs,
            'tfidf_matrix': self.tfidf_matrix
        }
        return d


# --------------------------
# get count matrix
# --------------------------


def get_count_matrix(docs: List[List[str]], ngram, hash_size, n_process):
    """Form a sparse word to document count matrix (inverted index).
    M[i, j] = # times word i appears in document j.
    """

    # Compute the count matrix in steps (to keep in memory)
    print('Mapping...')
    row, col, data = [], [], []

    with Pool(n_process) as pool:
        args = [(doc, doc_idx, ngram, hash_size)
                for doc_idx, doc in enumerate(docs)]
        imap_iter = pool.imap(unpack_count, args)
        for sub_row, sub_col, sub_data in tqdm(imap_iter, total=len(docs)):
            row.extend(sub_row)
            col.extend(sub_col)
            data.extend(sub_data)

    print('Creating sparse matrix...')
    count_matrix = scipy.sparse.csr_matrix(
        (data, (row, col)), shape=(hash_size, len(docs))
    )
    count_matrix.sum_duplicates()
    return count_matrix


def count(doc: List[str], doc_idx, ngram, hash_size):
    """Fetch the text of a document and compute hashed ngrams counts."""

    # Get ngrams from tokens
    ngrams = core.get_ngrams(n=ngram, tokens=doc)

    # Hash ngrams and count occurences
    counts = Counter([core.hash_murmurhash(gram, hash_size)
                      for gram in ngrams])

    # Return in sparse matrix data format.
    row = list(counts.keys())
    col = [doc_idx] * len(counts)
    data = list(counts.values())
    return row, col, data


def unpack_count(args):
    return count(*args)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.
    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = scipy.sparse.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


def main():
    # Register texts to Documents
    documents = SimpleDocuments()
    path = 'data/text/wiki_00.txt'
    n_lines = core.count_lines(path)
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=n_lines)):
            line = line.rstrip()
            if line == '':
                continue
            documents.add(line)

    documents.to_json('data/qas/documents.json')
    print('save documents as json')

    tokenizer = SentencePieceTokenizer('data/qas/spm.8000.model', 'id')
    texts = tokenizer.tokenize(documents.get_texts())
    indexer = Indexer()
    indexer.doc_to_tfidf_matrix(texts, n_process=8)
    index_data = indexer.to_dict()
    core.Pickle.pickle(index_data, 'data/qas/index.8000.pkl')


def main_from_encoded_text():
    with open('path', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    texts = [text.split() for text in texts]
    indexer = Indexer()
    tfidf_matrix = indexer.doc_to_tfidf_matrix(texts, n_process=8)
    print('tfidf_matrix', tfidf_matrix)


if __name__ == '__main__':
    print(__package__)
    main()
