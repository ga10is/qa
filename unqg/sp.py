import argparse

import sentencepiece as spm


# def spm_train(text_path, vocab_size=8000):
def spm_train(args):
    arg = '--input=%s --model_prefix=%s '\
        '--vocab_size=%d --character_coverage=%f'\
        ' --model_type=%s --user_defined_symbols=%s' % (
            args.input, args.model_prefix,
            args.vocab_size, args.character_coverage,
            args.model_type, args.user_defined_symbols)
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
        self.out_type = out_type

    def tokenize(self, text):
        tokens = self.sp.encode(text, out_type=self.out_type)
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
