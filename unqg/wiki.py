import sys
import os
import json
from multiprocessing import Pool

from tqdm import tqdm

from .logger import create_logger, get_logger


def extract_text(input_path, output_path):
    if os.path.isfile(output_path):
        return

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            data.append(text)

    text = '\n'.join(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
        get_logger().info('Done: %s' % output_path)


def extract_text_wrapper(args):
    return extract_text(*args)


def extract_texts(n_proc, input_dir, output_dir):

    file_list = os.listdir(input_dir)
    input_paths = [os.path.join(input_dir, file) for file in file_list]
    output_paths = [os.path.join(output_dir, file + '.txt')
                    for file in file_list]

    with Pool(n_proc) as pool:
        args = list(zip(input_paths, output_paths))
        imap = pool.imap(extract_text_wrapper, args)
        list(tqdm(imap, total=len(file_list)))


if __name__ == '__main__':
    # wiki()
    # tokenize()
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_proc = sys.argv[3]

    extract_texts(input_dir, output_dir, n_proc)
