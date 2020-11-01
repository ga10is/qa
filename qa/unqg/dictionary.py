import os
import argparse

from tqdm import tqdm
import sentencepiece as spm
import torch

"""
The class that implements the same methods as UnsupervisedMT/NMT/src/data/dictionary.py
"""

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'


class Dictionary:

    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.sp)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.sp.IdToPiece(i)

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return self.sp.PieceToId(w) != self.sp.unk_id()

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        if len(self.sp) != len(y):
            return False
        return all(self.sp.IdToPiece(i) == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        pass

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        assert not no_unk
        return self.sp[word]

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        It is not necessary for SentencePiece.
        """
        pass

    @staticmethod
    def index_data(path, bin_path, dico):

        if os.path.isfile(bin_path):
            print("Loading data from %s ..." % bin_path)
            data = torch.load(bin_path)
            assert dico == data['dico']
            return data

        positions = []
        sentences = []
        unk_words = {}

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                s = line.rstrip()
                if len(s) == 0:
                    continue

                indexed = dico.sp.encode(s)

                positions.append(
                    [len(sentences), len(sentences) + len(indexed)])
                sentences.extend(indexed)
                sentences.append(-1)

        positions = torch.LongTensor(positions)
        sentences = torch.LongTensor(sentences)
        data = {
            'dico': dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
        }
        print("Saving the data to %s ..." % bin_path)
        torch.save(data, bin_path)

        return data

    @staticmethod
    def encode(args):
        dico = Dictionary(args.model_file)

        Dictionary.index_data(args.input, args.bin_path, dico)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--bin_path', type=str, required=True)

    args = parser.parse_args()
    Dictionary.encode(args)
