from abc import ABCMeta, abstractmethod
import json

from . import core


class Documents(metaclass=ABCMeta):

    @abstractmethod
    def get_text_by_idx(self, idx):
        raise NotImplementedError('get_text_by_idx method is not implemented.')

    @abstractmethod
    def get_texts(self):
        raise NotImplementedError('get_texts method is not implemented.')

    @staticmethod
    @abstractmethod
    def load(path):
        raise NotImplementedError('load method is not implemented.')


class SimpleDocuments(Documents):

    def __init__(self):
        super().__init__()
        self.docs = {}
        self.doc_ids = []

    def add(self, text):
        doc_id = core.hash_sha1(text)
        self.doc_ids.append(doc_id)
        self.docs[doc_id] = text

    def get_text_by_idx(self, idx):
        doc_id = self.doc_ids[idx]
        return self.docs[doc_id]

    def get_text_by_id(self, doc_id):
        return self.docs[doc_id]

    def get_texts(self):
        texts = [self.docs[doc_id] for doc_id in self.doc_ids]
        return texts

    def to_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            doc_dict = {
                'docs': self.docs,
                'doc_ids': self.doc_ids
            }
            json.dump(doc_dict, f, ensure_ascii=False)

    @staticmethod
    def load(path):
        with open(path, 'r', encoding='utf-8') as f:
            doc_dict = json.load(f)
        documents = SimpleDocuments()
        documents.docs = doc_dict['docs']
        documents.doc_ids = doc_dict['doc_ids']
        return documents
