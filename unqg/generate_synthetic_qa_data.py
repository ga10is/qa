import argparse
from multiprocessing import Pool

import spacy
from tqdm import tqdm

from .parsers_and_writers import parse_paragraphs_from_txt
from .generate_clozes import generate_clozes_from_paragraph, named_entity_answer_generator as ne_answer_gen, \
    noun_phrase_answer_generator as np_answer_gen, is_appropriate_squad_datapoint


def generate_synthetic_training_data(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        paragraphs = parse_paragraphs_from_txt(f)
        paragraphs = list(paragraphs)

    print('=' * 50)
    print(f'Parsed {len(paragraphs)} paragraphs from {args.input_file}')
    print('=' * 50)

    # Create cloze
    answer_generator = ne_answer_gen if args.use_named_entity_clozes else np_answer_gen
    clozes = [c for p in paragraphs for c in generate_clozes_multiprocess(
        p, answer_generator)]
    # check_ne(paragraphs)


def generate_clozes_from_paragraph_wrapper(args):
    return generate_clozes_from_paragraph(*args)


def generate_clozes_multiprocess(paragraphs, answer_generator, n_proc):
    with Pool(n_proc) as pool:
        args = [(p, answer_generator) for p in paragraphs]
        imap = pool.imap(generate_clozes_from_paragraph_wrapper, args)
        clozes = list(tqdm(imap, total=len(paragraphs)))
        return clozes


def check_ne(paragraphs):
    nlp = spacy.load('ja_ginza')
    ents = set()

    for paragraph in paragraphs:
        if len(paragraph.text) > 20:
            doc = nlp(paragraph.text)
            print([t.ent_kb_id_ for t in doc])
            break
    '''
    for i, paragraph in enumerate(tqdm(paragraphs)):
        if i > 1000:
            break
        para_doc = nlp(paragraph.text)
        for sentence in para_doc.sents:
            ents = ents | {e.label_ for e in sentence.ents}
    print(ents)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for extractive QA tasks without supervision')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument("--use_named_entity_clozes", action='store_true')

    args = parser.parse_args()
    generate_synthetic_training_data(args)
