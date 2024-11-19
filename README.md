```
japanese-lipnet/
├── data/
│   ├── raw/                        # オリジナルのROHANデータ
│   │   ├── videos/
│   │   │   ├── LFROI_ROHAN4600_0001.mp4
│   │   │   ├── LFROI_ROHAN4600_0002.mp4
│   │   │   └── ...
│   │   ├── lab/
│   │   │   ├── ROHAN4600_0001.lab
│   │   │   ├── ROHAN4600_0002.lab
│   │   │   └── ...
│   │   └── landmarks/
│   │       ├── ROHAN4600_0001_points.csv
│   │       ├── ROHAN4600_0002_points.csv
│   │       └── ...
│   │
│   ├── processed/                  # 前処理済みデータ
│   │   ├── train/
│   │   │   ├── videos/
│   │   │   ├── lab/
│   │   │   └── landmarks/
│   │   ├── val/
│   │   │   ├── videos/
│   │   │   ├── lab/
│   │   │   └── landmarks/
│   │   └── test/
│   │   │   ├── videos/
│   │   │   ├── lab/
│   │   │   └── landmarks/
│   │   │
│   │   │── train_list.txt  # データ分割情報
│   │   ├── val_list.txt
│   │   └── test_list.txt
│   │
│   ├── preprocess/
│       └── prepare_dataset.py  # データセット前処理
│   
├── models/
│   ├── conformer.py      # Conformerモデル実装
│   ├── frontend.py       # 画像特徴抽出
│   └── japanese_lipnet.py # メインモデル
│
├── utils/
│   ├── metrics.py        # CER/WER計算
│   └── japanese_tokenizer.py  # 日本語音素処理
│
├── configs/
│   └── config.py         # 設定ファイル
├── dataset.py            # データローダー
├── train.py             # 学習スクリプト
└── demo.py              # デモスクリプト
```