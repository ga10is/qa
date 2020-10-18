import argparse
import spacy

from .generate_clozes import named_entity_answer_generator, noun_phrase_answer_generator, \
    is_appropriate_cloze, is_appropriate_answer, mask_answer


def mask_text(args):
    nlp = spacy.load('ja_ginza')

    answer_generator = named_entity_answer_generator if args.use_named_entity_clozes else noun_phrase_answer_generator
    with open(args.input_file, 'r', encoding='utf-8') as f:
        masked_sents = [masked for line in f
                        for masked in mask_line_generator(line, nlp, answer_generator)]

    print('\n'.join(masked_sents))
    # with open(args.output_file, 'w', encoding='utf-8') as f:
    # f.writelines(masked_sents)


def mask_line_generator(line, nlp, answer_generator):
    doc = nlp(line)
    for sentence in doc.sents:
        if is_appropriate_cloze(sentence.text):
            answers = answer_generator(sentence)
            for answer_text, answer_start, answer_type in answers:
                if is_appropriate_answer(answer_text):
                    cloze_text = mask_answer(
                        sentence.text, answer_text, answer_start, answer_type)
                    yield cloze_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for extractive QA tasks without supervision')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument("--use_named_entity_clozes", action='store_true')

    args = parser.parse_args()
    mask_text(args)
