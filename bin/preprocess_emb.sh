set -e

cd `dirname $0`
cd ../

# Data preprocessing configuration
N_PROCESSES=8
CODES=60000
N_THREADS=48
N_EPOCHS=10
MASK_LIST=IDENTITYMASK,PLACEMASK,THINGMASK,TEMPORALMASK,NUMERICMASK

# Initialize tools and data paths
WORK_PATH=$PWD
DATA_PATH=$PWD/data
TOOLS_PATH=$PWD/tools
MONO_PATH=$PWD/mono

echo "Working directory: $WORK_PATH"

mkdir -p $TOOLS_PATH
mkdir -p $MONO_PATH

# wiki
# WIKI_TEXT_PATH=$DATA_PATH/wiki.txt
WIKI_TEXT_PATH=$DATA_PATH/text/wiki_00.txt

# SentencePiece
SP_PY_PATH=$WORK_PATH/unqg/sp.py
SP_VOCAB=$MONO_PATH/spm.$CODES
SP_ENCODED_TEXT=$MONO_PATH/encoded_wiki.$CODES.txt

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext



# Install SentencePiece
if pip freeze | grep "sentencepiece" >/dev/null; then
    echo "Already installed SentencePiece."
else
    echo "Installing SentencePiece..."
    pip install sentencepiece
fi

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


# Exec SentencePiece
cd $WORK_PATH
if [ ! -f "$WIKI_TEXT_PATH" ]; then
    echo "Execute bin/preprocess_wiki.sh"
fi

# train vocabulary
if ! [[ -f "$SP_VOCAB.vocab" && -f "$SP_VOCAB.model" ]]; then
    echo "Extracting vocabulary..."
    python $SP_PY_PATH train --input $WIKI_TEXT_PATH --model_prefix $SP_VOCAB --vocab_size=$CODES --user_defined_symbols=$MASK_LIST
fi
echo "Extracted vocabulary: $SP_VOCAB.vocab and $SP_VOCAB.model"

# encode text to index
if [ ! -f "$SP_ENCODED_TEXT" ]; then
    echo "Encoding text..."
    python $SP_PY_PATH encode --input $WIKI_TEXT_PATH --output $SP_ENCODED_TEXT --model $SP_VOCAB.model --output_format id
fi
echo "Encoded text to index in : $SP_ENCODED_TEXT"

# spm_train
# spm_encode
# spm_decode
# spm_export_vocab



# Exec fastText