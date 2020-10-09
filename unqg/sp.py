import sys

import sentencepiece as spm


def spm_train(text_path, vocab_size=8000):
    arg = '--input=%s --model_prefix=sp --vocab_size=%d --character_coverage=0.9995 --model_type=unigram' % (
        text_path, vocab_size)
    spm.SentencePieceTrainer.Train(arg)


if __name__ == '__main__':
    text_path = sys.argv[1]

    spm_train(text_path, 8000)
