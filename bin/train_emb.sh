set -e

cd `dirname $0`
cd ../

# Data preprocessing configuration
N_THREADS=48
N_EPOCHS=10

# Initialize tools and data paths
WORK_PATH=$PWD
DATA_PATH=$PWD/data
TOOLS_PATH=$PWD/tools

mkdir -p $TOOLS_PATH

echo "Working directory: $WORK_PATH"

# Input text, whose words are separated with spaces
ENCODED_TEXT=$DATA_PATH/mono/encoded_concat.60000.txt

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

#
# Install tools
#

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


#
# Train embeddings
#

# Check required file
if [ ! -f "$ENCODED_TEXT" ]; then
    echo "Encoded text file is not exist in: $ENCODED_TEXT"
    exit 1
fi

# fastText
if ! [[ -f "$ENCODED_TEXT.vec" ]]; then
    echo "Training fastText on $ENCODED_TEXT..."
    $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $ENCODED_TEXT -output $ENCODED_TEXT
fi
echo "Output embeddings to: $ENCODED_TEXT.vec"
