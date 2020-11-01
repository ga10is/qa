import argparse

import sentencepiece as spm


# def spm_train(text_path, vocab_size=8000):
def spm_train(args):
    arg_list = []
    arg_list.append('--input=%s' % args.input)
    arg_list.append('--model_prefix=%s' % args.model_prefix)
    arg_list.append('--vocab_size=%d' % args.vocab_size)
    arg_list.append('--character_coverage=%f' % args.character_coverage)
    arg_list.append('--model_type=%s' % args.model_type)
    arg_list.append('--shuffle_input_sentence=%s' %
                    args.shuffle_input_sentence)
    if args.input_sentence_size is not None:
        arg_list.append('--input_sentence_size=%d' % args.input_sentence_size)
    arg = ' '.join(arg_list)
    spm.SentencePieceTrainer.Train(arg)


def spm_encode(args):
    out_type = int if args.output_format == 'id' else str
    sp = spm.SentencePieceProcessor(model_file=args.model)

    # TODO: devide reading
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    encoded_lines = sp.encode(lines, out_type=out_type)
    encoded_lines = [' '.join(map(str, L)) for L in encoded_lines]

    with open(args.output, 'w', encoding='utf-8') as f:
        f.writelines(encoded_lines)


class SentencePieceTokenizer:
    def __init__(self, model_file, out_type):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.out_type = int if out_type == 'id' else str

    def tokenize(self, text):
        """
        Parameters
        ----------
        text: str or list of str

        Return
        ------
        tokens list
            list of str if text argument is str,
            list of list of str if text argument is list of str
        """
        tokens = self.sp.encode(text, out_type=self.out_type)
        if self.out_type == int:
            tokens = [str(num) for num in tokens]
        return tokens


def spm_decode(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SentencePiece')
    subparsers = parser.add_subparsers()

    # spm_train command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--input')
    parser_train.add_argument('--model_prefix')
    parser_train.add_argument('--vocab_size', type=int)
    parser_train.add_argument('--character_coverage',
                              default=0.9995, type=float)
    parser_train.add_argument(
        '--model_type', default='unigram', choices=['unigram', 'bpe', 'char', 'word'])
    parser_train.add_argument('--user_defined_symbols', type=str, default='')
    parser_train.add_argument('--input_sentence_size', type=int)
    parser_train.add_argument('--shuffle_input_sentence',
                              choices=['true', 'false'], default='true')
    parser_train.set_defaults(handler=spm_train)

    # spm_encode command
    parser_encode = subparsers.add_parser('encode')
    parser_encode.add_argument('--input')
    parser_encode.add_argument('--output')
    parser_encode.add_argument('--model')
    parser_encode.add_argument('--output_format', choices=['id', 'piece'])
    parser_encode.set_defaults(handler=spm_encode)

    args = parser.parse_args()
    args.handler(args)
