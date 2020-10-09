# Unsupervised Question Generation
## 前処理
- BPE(Sentencepieceで)
- preprocess.py
- fastText-> .vecファイル

質問文の前処理は上でいいのはわかった。
Cloze taskしたwikiの文はどのようにしてSentencepieceにかけるのか？
マスクした語句をどうすればいいのか？

wiki->Clozeな文の集合↓
質問文の集合↓
合わせた集合->

学習
- データ準備
    - MASKしたwiki+質問文
- unmt_translation.preprocess()
    - データをdump
    - sentencepiece
    - fasttext
- train

推論
- データ準備
    - MASKしたwiki
- unmt_translation.preprocess()
    - データをdump
    - sentencepiece
    - binarize_data -> Dictionary
- unmt_translation.perform_translation()
    - src.model.build_mt_model()
        - build_attention_model()
    - fasttext
- infer

### wiki
wikiのダウンロード
```
curl https://dumps.wikimedia.org/jawiki/20201001/jawiki-20201001-pages-articles-multistream.xml.bz2 -o jawiki-20201001-pages-articles-multistream.xml.bz2
```

wikiextractorのインストール
```
pip install wikiextractor
```

wikiextractorの実行
```
python -m wikiextractor.WikiExtractor -o ../data2/extracted -b 100M --json --processes 8 ../data2/jawiki-latest-pages-articles.xml.bz2
```
INFO: 4224210   ハーバート男爵
INFO: 4224211   メリッサ・ブランネン
INFO: 4224212   フランセ (曖昧さ回避)
INFO: Finished 8-process extraction of 1230899 articles in 2030.8s (606.1 art/s)
INFO: total of page: 1783966, total of articl page: 1230899; total of used articl page: 1230899
