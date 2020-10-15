set -e

cd `dirname $0`
cd ../


WORK_PATH=$PWD
DATA_PATH=$PWD/data
TOOLS_PATH=$PWD/tools

# Common Crawl
CCRAWL_PATH=$DATA_PATH/ccrawl
CCRAWL_INDEX_PATH=$CCRAWL_PATH/index
CCRAWL_QUESTION_TEXT=$CCRAWL_PATH/question.txt

# QA site url
CRAWL_TARGET_URL="detail.chiebukuro.yahoo.co.jp/*"

echo "Working directory: $WORK_PATH"

# make directory
mkdir -p $TOOLS_PATH
mkdir -p $CCRAWL_INDEX_PATH

# cdx-index-client
# https://github.com/ikreymer/cdx-index-client
INDEX_CLIENT_DIR=$TOOLS_PATH/cdx-index-client
INDEX_CLIENT_PY_PATH=$INDEX_CLIENT_DIR/cdx-index-client.py

# Download sub WARC file and extract question script
EXTRACT_QUESTION_PY_PATH=$WORK_PATH/ccrawl/crawl.py

# Download cdx-index-client
cd $TOOLS_PATH
if [ ! -d "$INDEX_CLIENT_DIR" ]; then
    echo "Cloning cdx-index-client from GitHub repository..."
    git clone https://github.com/ikreymer/cdx-index-client.git

    echo "Installing dependent libraries..."
    conda install --yes --file $INDEX_CLIENT_DIR/requirements.txt
fi
echo "Cloned cdx-index-clinet in:" $INDEX_CLIENT_DIR

# Search index of chiebukuro pages
if ls $CCRAWL_INDEX_PATH/prefix* > /dev/null 2>&1; then
    echo "Already downloaded index of pages"
else
    echo "Searching index of chiebukuro pages..."
    python $INDEX_CLIENT_PY_PATH -d $CCRAWL_INDEX_PATH --json $CRAWL_TARGET_URL
fi
echo "Downloaded index of pages in:" $CCRAWL_INDEX_PATH

# Download sub WARC file and extract question sentences from the file
if [ ! -f "$CCRAWL_QUESTION_TEXT" ]; then
    echo "Download sub WARC file and extract question..."
    python $EXTRACT_QUESTION_PY_PATH $CCRAWL_INDEX_PATH $CCRAWL_QUESTION_TEXT
fi
echo "Created question file in :" $CCRAWL_QUESTION_TEXT
