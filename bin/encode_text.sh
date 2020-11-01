set -e

cd `dirname $0`
cd ../

# Data preprocessing configuration
N_PROCESSES=8
CODES=60000
N_EPOCHS=10
MASK_LIST=IDENTITYMASK,PLACEMASK,THINGMASK,TEMPORALMASK,NUMERICMASK

# Initialize tools and data paths
WORK_PATH=$PWD
DATA_PATH=$PWD/data
TOOLS_PATH=$PWD/tools
MONO_PATH=$DATA_PATH/mono

echo "Working directory: $WORK_PATH"

mkdir -p $TOOLS_PATH
mkdir -p $MONO_PATH

# wiki/question/concat text
# WIKI_TEXT_PATH=$DATA_PATH/wiki.txt
# WIKI_TEXT_PATH=$DATA_PATH/text/wiki_00.txt
WIKI_TEXT_PATH=$DATA_PATH/wiki/mask_wiki.txt
QUESTION_TEXT_PATH=$DATA_PATH/ccrawl/question.txt
CONCAT_TEXT_PATH=$MONO_PATH/concat.txt

# SentencePiece
SP_PY_PATH=$WORK_PATH/qa/unqg/sp.py
SP_VOCAB=$MONO_PATH/spm.$CODES
ENCODED_TEXT=$MONO_PATH/encoded_concat.$CODES.txt

# Dictionary
DIC_PY_PATH=$WORK_PATH/qa/unqg/dictionary.py
ENCODED_WIKI_BIN=$MONO_PATH/encoded_wiki.$CODES.pth
ENCODED_QUESTION_BIN=$MONO_PATH/encoded_question.$CODES.pth

#
# Install tools
#

# Install SentencePiece
if pip freeze | grep "sentencepiece" >/dev/null; then
    echo "Already installed SentencePiece."
else
    echo "Installing SentencePiece..."
    pip install sentencepiece
fi

#
# Encoding text to code
#

# Exec SentencePiece
cd $WORK_PATH
if [ ! -f "$WIKI_TEXT_PATH" ]; then
    echo "Execute bin/downlaod_wiki.sh and bin/mask_wiki.sh"    
    exit 1
fi
if [ ! -f "$QUESTION_TEXT_PATH" ]; then
    echo "Execute bin/download_question.sh"
    exit 1
fi

# Concatenate wiki text and question text
if [ ! -f "$CONCAT_TEXT_PATH" ]; then
    echo "Concatenating wiki and question text..."
    cat $WIKI_TEXT_PATH $QUESTION_TEXT_PATH | shuf > $CONCAT_TEXT_PATH
fi
echo "Concatenated text in: $CONCAT_TEXT_PATH"

# Train vocabulary
if ! [[ -f "$SP_VOCAB.vocab" && -f "$SP_VOCAB.model" ]]; then
    echo "Extracting vocabulary..."
    python $SP_PY_PATH train --input $CONCAT_TEXT_PATH --model_prefix $SP_VOCAB --vocab_size=$CODES --user_defined_symbols=$MASK_LIST
fi
echo "Extracted vocabulary: $SP_VOCAB.vocab and $SP_VOCAB.model"

# Encode text to code
if [ ! -f "$ENCODED_TEXT" ]; then
    echo "Encoding text to code..."
    python $SP_PY_PATH encode --input $CONCAT_TEXT_PATH --output $ENCODED_TEXT --model $SP_VOCAB.model --output_format id    
fi
echo "Encoded text in: $ENCODED_TEXT"

# Encode text to binary file
if ! [[ -f "$ENCODED_WIKI_BIN" &&  -f "$ENCODED_QUESTION_BIN" ]]; then
    echo "Encoding wiki text to binary file..."
    python $DIC_PY_PATH --input $WIKI_TEXT_PATH --bin_path $ENCODED_WIKI_BIN --model_file $SP_VOCAB.model
    python $DIC_PY_PATH --input $QUESTION_TEXT_PATH --bin_path $ENCODED_QUESTION_BIN --model_file $SP_VOCAB.model
fi
echo "Binarized wiki file in : $ENCODED_WIKI_BIN"
echo "Binarized question file in : $ENCODED_QUESTION_BIN"
