import argparse
import spacy

from .generate_clozes import named_entity_answer_generator, noun_phrase_answer_generator, \
    is_appropriate_cloze, is_appropriate_answer, mask_answer

from functools import wraps
import time


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}: {elapsed_time}[s]")
        return result
    return wrapper


@stop_watch
def mask_text(args):
    print('Loading spacy model...')
    nlp = spacy.load('ja_ginza')

    answer_generator = named_entity_answer_generator if args.use_named_entity_clozes else noun_phrase_answer_generator

    subline_gen = subline_generator(args.input_file, 1000)
    for i, sublines in enumerate(subline_gen):

        masked_sents = list(mask_texts_generator(sublines,
                                                 nlp,
                                                 answer_generator,
                                                 n_process=8))
        with open(args.output_file, 'a', encoding='utf-8') as f:
            f.writelines('\n'.join(masked_sents))


def subline_generator(path, n_lines, start_line=0):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sublines = []
        for i, line in enumerate(f):
            if i < start_line:
                # skip this line
                continue
            sublines.append(line)
            if (i + 1) % n_lines == 0:
                yield sublines
                sublines.clear()
        if len(sublines) > 0:
            yield sublines


def mask_texts_generator(texts, nlp, answer_generator, n_process):
    for doc in nlp.pipe(texts,
                        disable=['CompoundSplitter', 'BunsetuRecognizer'],
                        n_process=n_process,
                        batch_size=100):
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
