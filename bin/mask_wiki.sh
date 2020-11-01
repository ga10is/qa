set -e

cd `dirname $0`
cd ../


# Initialize tools and data paths
WORK_PATH=$PWD
DATA_PATH=$PWD/data
TOOLS_PATH=$PWD/tools
MONO_PATH=$DATA_PATH/mono

WIKI_DIR=$DATA_PATH
# WIKI_TEXT_PATH=$WIKI_DIR/wiki.txt
WIKI_TEXT_PATH=$WIKI_DIR/text/wiki_01.txt
MASKED_WIKI_TEXT_PATH=$WIKI_DIR/mask_wiki.txt

# Program
# MASK_PY_PATH=$WORK_PATH/unqg/mask_text.py

echo "Working directory: $WORK_PATH"

if [ ! -f "$WIKI_TEXT_PATH" ]; then
    echo "Not exist source wiki file: $WIKI_TEXT_PATH"
    exit 1
fi

cd $WORK_PATH
# Mask wiki text
if [ ! -f "$MASKED_WIKI_TEXT_PATH" ]; then
    echo "Masking wiki text..."
    python -m qa.unqg.mask_text --input_file $WIKI_TEXT_PATH --output_file $MASKED_WIKI_TEXT_PATH --use_named_entity_clozes
fi
echo "Masked wiki text in: $MASKED_WIKI_TEXT_PATH"
