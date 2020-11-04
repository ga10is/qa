import argparse
import unicodedata
import re
import spacy
from tqdm import tqdm
from functools import wraps
import time
from multiprocessing import Pool
import itertools
from collections import deque

from .generate_clozes import named_entity_answer_generator, noun_phrase_answer_generator, \
    is_appropriate_cloze, is_appropriate_answer, mask_answer


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
        print('Processing text...')
        sublines = [processed for processed in preprocess_generator(sublines)]
        print('Dividing text to subclause...')
        subclauses = [
            subclause for subclause in subclauses_generator_from_texts(sublines, nlp)]
        masked_sents = list(mask_texts_generator(subclauses,
                                                 nlp,
                                                 answer_generator,
                                                 n_process=-1))
        with open(args.output_file, 'a', encoding='utf-8') as f:
            f.writelines('\n'.join(masked_sents))


def subline_generator(path, n_sublines, start_line=0):
    n_lines = count_lines(path)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sublines = []
        for i, line in enumerate(tqdm(f, total=n_lines)):
            if i < start_line:
                # skip this line
                continue

            sublines.append(line)
            if (i + 1) % n_sublines == 0:
                yield sublines
                sublines.clear()
        if len(sublines) > 0:
            yield sublines


def preprocess_generator(texts):
    for text in texts:
        text = text.rstrip()
        if text == '':
            continue
        if contains_diacritical_mark(text):
            # skip text which contains diacritical marks
            continue
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'(\d+年)\(.*?\d+年\)', '\\1', text)
        yield text


def contains_diacritical_mark(text):
    diacritaical_marks = ['\u00B4', '\u00A8', '\u203E',
                          '\u00AF', '\uFFE3', '\u00B8', '\u309B', '\u309C']
    for mark in diacritaical_marks:
        if mark in text:
            return True
    return False

#
# Subclause
#


def subclauses_generator_from_texts(texts, nlp):
    for text in texts:
        for subclause in subclauses_generator(text, nlp):
            yield subclause


def subclauses_generator(text, nlp):
    doc = nlp(text)
    for tokens in doc.sents:
        exclude_indices = []
        for token in tokens:
            # 1つもadvclがないときの対応
            if token.dep_ == 'advcl' and token.pos_ != 'AUX':
                subclause, indices = get_subtree(
                    tokens, token.i, exclude_indices)
                exclude_indices.extend(indices)
                yield subclause
        main_clause = ''.join([token.orth_ for token in tokens
                               if token.i not in exclude_indices])
        if main_clause != '':
            yield main_clause


def get_subtree(tokens, subtree_root_idx, exclude):
    """
    Returns
    -------
    subcluase text
        str
    """
    indices = []
    queue = deque()
    queue.append(subtree_root_idx)
    while len(queue) > 0:
        head_idx = queue.popleft()
        for token in tokens:
            if token.i in exclude:
                continue
            if token.head.i == head_idx:
                indices.append(token.i)
                exclude.append(token.i)
                queue.append(token.i)
            elif token.i == head_idx:
                indices.append(token.i)

    indices.sort()
    subclause = ''.join([t.orth_ for t in tokens if t.i in indices])
    return subclause, indices


def count_lines(path, blocksize=65536):
    def blocks(f):
        while True:
            b = f.read(blocksize)
            if b:
                yield b
            else:
                break

    with open(path, 'r') as f:
        return sum(bl.count('\n') for bl in blocks(f))

#
# Mask text
#


def mask_texts_generator(texts, nlp, answer_generator, n_process):
    for doc in nlp.pipe(texts,
                        disable=['CompoundSplitter', 'BunsetuRecognizer'],
                        n_process=n_process,
                        batch_size=100):
        for sentence in doc.sents:
            if is_appropriate_cloze(sentence):
                answers = answer_generator(sentence)
                for answer_text, answer_start, answer_type in answers:
                    if is_appropriate_answer(answer_text):
                        cloze_text = mask_answer(
                            sentence.text, answer_text, answer_start, answer_type)
                        if '\n' in cloze_text:
                            print(cloze_text)
                        yield cloze_text


def mask_text_to_cloze_wrapper(args):
    return mask_text_to_cloze(*args)


def mask_text_to_cloze(text, answer_generator):
    try:
        doc = nlp(text, disable=['CompoundSplitter', 'BunsetuRecognizer'])

        cloze_texts = []
        for sentence in doc.sents:
            if is_appropriate_cloze(sentence.text):
                answers = answer_generator(sentence)
                for answer_text, answer_start, answer_type in answers:
                    if is_appropriate_answer(answer_text):
                        cloze_text = mask_answer(
                            sentence.text, answer_text, answer_start, answer_type)
                        if '\n' in cloze_text:
                            print(cloze_text)
                        cloze_texts.append(cloze_text)
        return cloze_texts
    except Exception as e:
        print(e)
        return []


def mask_texts_to_cloze_pipe(texts, answer_generator):
    try:
        cloze_texts = []
        for doc in nlp.pipe(texts,
                            disable=['CompoundSplitter', 'BunsetuRecognizer'],
                            batch_size=100):
            for sentence in doc.sents:
                if is_appropriate_cloze(sentence.text):
                    answers = answer_generator(sentence)
                    for answer_text, answer_start, answer_type in answers:
                        if is_appropriate_answer(answer_text):
                            cloze_text = mask_answer(
                                sentence.text, answer_text, answer_start, answer_type)
                            if '\n' in cloze_text:
                                print(cloze_text)
                            cloze_texts.append(cloze_text)
        return cloze_texts
    except Exception as e:
        print(e)
        return []


def mask_texts_to_cloze(texts, answer_generator, n_process):
    with Pool(n_process) as pool:
        args = [(text, answer_generator) for text in texts]
        imap_iter = pool.imap(mask_text_to_cloze_wrapper, args)
        result = list(imap_iter)
        result = list(itertools.chain.from_iterable(result))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for extractive QA tasks without supervision')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument("--use_named_entity_clozes", action='store_true')

    args = parser.parse_args()
    mask_text(args)
