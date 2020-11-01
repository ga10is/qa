set -e

cd `dirname $0`
cd ../

# Data preprocessing configuration
CODES=8000

# Initialize tools and data paths
WORK_PATH=$PWD
DATA_PATH=$PWD/data
INPUT_TEXT=$1

# SentencePiece
SP_PY_PATH=$WORK_PATH/qa/unqg/sp.py
SP_VOCAB=$DATA_PATH/spm.$CODES

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

# Train vocabulary
if ! [[ -f "$SP_VOCAB.vocab" && -f "$SP_VOCAB.model" ]]; then
    echo "Extracting vocabulary..."
    python $SP_PY_PATH train --input $INPUT_TEXT --model_prefix $SP_VOCAB --vocab_size=$CODES
fi
echo "Extracted vocabulary: $SP_VOCAB.vocab and $SP_VOCAB.model"
