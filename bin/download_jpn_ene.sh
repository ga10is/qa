set -e

cd `dirname $0`
cd ../

WORK_PATH=$PWD
DATA_PATH=$PWD/data

NEN_PREFIX=$DATA_PATH/ja_ene
NEN_TOOL_PATH=$WORK_PATH/ne/extract_ene.py

if [ ! -f "$NEN_PATH" ]; then
    echo "Generating extended named entities in japanese..."
    python $NEN_TOOL_PATH $NEN_PREFIX
fi
echo "Extended named entities in: $NEN_PREFIX.json"
