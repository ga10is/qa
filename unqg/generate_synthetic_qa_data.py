import argparse

from .parsers_and_writers import parse_paragraphs_from_txt
from .generate_clozes import generate_clozes_from_paragraph, named_entity_answer_generator as ne_answer_gen, \
    noun_phrase_answer_generator as np_answer_gen, is_appropriate_squad_datapoint


def generate_synthetic_training_data(args):
    with open(args.input_file) as f:
        paragraphs = parse_paragraphs_from_txt(f)
    paragraphs = list(paragraphs)

    print('=' * 50)
    print(f'Parsed {len(paragraphs)} paragraphs from {args.input_file}')
    print('=' * 50)

    # Create cloze
    answer_generator = ne_answer_gen if args.use_named_entity_clozes else np_answer_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for extractive QA tasks without supervision')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument("--use_named_entity_clozes", action='store_true')

    args = parser.parse_args()
    generate_synthetic_training_data(args)
