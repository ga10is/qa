#!/bin/sh

set -e

cd `dirname $0`
cd ../

echo $PWD

N_PROCESSES=8
CODES=60000
N_THREADS=48
N_EPOCHS=10

WORK_PATH=$PWD
DATA_PATH=$PWD/data

WIKI_EXTRACT_PATH=$DATA_PATH/extracted
WIKI_TEXTDIR_PATH=$DATA_PATH/text
WIKI_TEXT_PATH=$DATA_PATH/wiki.txt

# Install wikiextractor
if pip freeze | grep "wikiextractor" >/dev/null; then
    echo "Already installed wikiextractor"
else
    echo "Installing wikiextractor..."
    pip install wikiextractor
fi

# Download wiki data
WIKI_URL=https://dumps.wikimedia.org/jawiki/20201001/jawiki-20201001-pages-articles-multistream.xml.bz2
WIKI_BZ2_PATH=$DATA_PATH/jawiki-20201001-pages-articles-multistream.xml.bz2

if [ ! -d "$WIKI_BZ2_PATH" ]; then
    echo "Downloading wiki data..."
    # curl $WIKI_URL -o $WIKI_BZ2_PATH
fi
echo "Downloaded wiki data: $WIKI_BZ2_PATH"

# Extract data in json with wikiextractor
if [ ! -d "$WIKI_EXTRACT_PATH" ]; then
    echo "Extracting wiki text in json with wikiextractor..."
    # python -m wikiextractor.WikiExtractor -o $WIKI_EXTRACT_PATH -b 100M --json --processes $N_PROCESSES $WIKI_BZ2
fi
echo "Extracted wiki data: $WIKI_EXTRACT_PATH"

# Extract text
if [ ! -d "$WIKI_TEXTDIR_PATH" ]; then
    echo "Extracting wiki texts from json files..."
    # python -m unqg.wiki $WIKI_EXTRACT_PATH/AA $WIKI_TEXTDIR_PATH $N_PROCESS
fi

# Concatnate wiki text
if [ ! -f "$WIKI_TEXT_PATH" ]; then
    echo "Concatnate wiki text files to 1 file.."
    # find $WIKI_TEXTDIR_PATH | grep wiki | awk 'system("cat "$0" >> "$WIKI_TEXT_PATH"")'
    cat $WIKI_TEXTDIR_PATH/wiki_*.txt > $WIKI_TEXT_PATH
fi
echo "Concatnated text: $WIKI_TEXT_PATH"






# question data

#if ! [[ -f "$CONCAT_BPE.vec" ]]; then
#  echo "Training fastText on $CONCAT_BPE..."
#  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
#fi
#echo "fastText embeddings in: $CONCAT_BPE.vec"

