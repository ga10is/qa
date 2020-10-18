import sys
import os
import json
import io
import re
import gzip
from multiprocessing import Pool

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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


def parse_index(dir_path):
    pages = []
    print(dir_path)
    index_files = os.listdir(dir_path)

    for index_file in index_files:
        index_path = os.path.join(dir_path, index_file)
        print('Index file: %s' % index_path)

        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                page = json.loads(line)
                if page['status'] == '200':
                    pages.append(page)
    return pages


def extract_question_from_subwarc(page):
    html = download_subwarc(page)
    text = html2text(html)

    sents = simple_sentence_segment(text)
    questions = extract_question(sents)

    return questions


def download_subwarc(result):
    """Donwload WARC"""
    offset, length = int(result['offset']), int(result['length'])
    offset_end = offset + length - 1

    try:
        url = 'https://commoncrawl.s3.amazonaws.com/%s' % result['filename']
        extract_range = 'bytes=%d-%d' % (offset, offset_end)
        response = requests.get(url,
                                headers={'Range': extract_range}
                                )
    except Exception as e:
        print('Error occurred: %s' % str(e))
        print('skip: %s' % url)
        return ''

    zipped_file = io.BytesIO(response.content)
    unzipped_file = gzip.GzipFile(fileobj=zipped_file)

    raw_data = unzipped_file.read()
    try:
        data: str = raw_data.decode("utf-8")
    except UnicodeDecodeError:
        print(f"Warning: Could not extract file downloaded from {url}")
        data = ""

    if len(data) > 0:
        data_parts = data.strip().split("\r\n\r\n", 2)
        result["html"] = data_parts[2] if len(data_parts) == 3 else ""

        html = data_parts[2]
        return html
    else:
        raise ValueError('Unexpected data in: %s' % url)


def html2text(html):
    soup = BeautifulSoup(html, 'html.parser')

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    p_texts = [c.get_text() for c in soup.find_all('p')]
    text = '\n'.join(p_texts)
    return text


def simple_sentence_segment(text):
    escape_char = "\u222f"
    punct = r"。!?！？"

    splitRegex = r"(?<!%s)([%s])(?!%s)" % (escape_char, punct, escape_char)
    result = re.sub(splitRegex, "\\1\n", text)
    unescapeRegex = r"(%s)([%s])(%s)" % (escape_char, punct, escape_char)
    result = re.sub(unescapeRegex, "\\2", result)
    sentences = [sent.strip() for sent in result.splitlines()]

    return sentences


def extract_question(sentences):
    q_sentences = set()
    for sentence in sentences:
        if endswith_question_mark(sentence) and contains_wh(sentence):
            q_sentences.add(sentence)

    return q_sentences


def endswith_question_mark(text):
    return text.endswith('?') or text.endswith('？')


def contains_wh(text):
    wh_words = ['だれ', '誰', '何', 'どこ', '何処', 'いつ', 'いくつ', 'いくら']
    notwh_words = ['何故', '如何', '何とか', '何も', '何だか', 'いつも', 'いつでも']

    def is_appropriate(text):
        for wh_word in wh_words:
            if wh_word in text:
                return True
        return False

    def not_appropriate(text):
        for notwh_word in notwh_words:
            if notwh_word in text:
                return True
        return False

    contains = is_appropriate(text) and not not_appropriate(text)
    return contains


def cleansing(sentence):

    return sentence


if __name__ == '__main__':

    index_dir_path = sys.argv[1]
    output = sys.argv[2]
    n_proc = 8

    pages = parse_index(index_dir_path)
    # pages = pages[:100]

    # main process
    with Pool(n_proc) as pool:
        imap = pool.imap(extract_question_from_subwarc, pages)
        result = list(tqdm(imap, total=len(pages)))
    # flatten
    q_sentences = set()
    for q_set in result:
        q_sentences = q_sentences.union(q_set)

    # cleansing
    q_sentences = [cleansing(sent) for sent in q_sentences]
    q_sentences = [sent for sent in q_sentences if sent is not None]

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(q_sentences))
