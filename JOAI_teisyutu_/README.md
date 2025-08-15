# JOAI 2025 解法
JOAI 2025 での最終提出 (Public LB: 0.993, Private LB: 0.996) を再現するためのコードです。ディレクトリ構造は以下のようになっています。
```
.
├── data  コンペのデータセットとデータ処理用コード
└── code /
    ├── exp0  exp0 のコードと予測・OOFのCSV
    ├── exp1 
    └── ...
```
# 実行環境
- # kaggle 環境 (exp0 ~ exp26, exp28)
    - T4 x2 付きの環境で実行。
    - 一部ローカルで実行したが同等の結果が得られることを確認している。
    - 必要ライブラリ等は requirements_kaggle.txt を参照。

- # RunPod 環境 (exp27)
    - A40 付きの環境で実行。
    - 必要ライブラリ等は requirements_runpod.txt を参照。

# 実行方法
1. データセットを以下のようにdata以下に配置する。
    ```
    .
    └── data/
        ├── images  新しく配置
        ├── train.csv  新しく配置
        ├── test.csv  新しく配置
        ├── sample_submission.csv  新しく配置
        ├── data_transform.py  元からあるコード
        ├── add_feature_test.py  元からあるコード
        ├── extract_temp_test.py  元からあるコード
        ├── add_feature_train.py  元からあるコード
        └── extract_temp_train.py  元からあるコード
    ```
2. rerun.shと同じ順に data_transform.py から exp26.py までを実行する。
3. exp27.py を実行する。
4. exp28.py を実行する。この時 exp28 のディレクトリに作成された submission.csv が最終提出とほとんど同じものになるはずです。

# 再現性について
GPUを使うコードは本番時は jupyter notebook 上で実行しました。1セルにコードをすべて入れていたので、実行結果は同じになると思っていましたが、同じコードでも一部結果が異なることがあった(原因不明)ので、より正確な再現のためには jupyter notebook で走らせることが望ましいと考えられます。  
また、他の箇所についても実行結果がわずかに異なる可能性があります。元から入っている exp*** の oof_predictions_probs.csv, test_predictions_probs.csvは実際にアンサンブルしたときのcsvです。これらを用いてアンサンブルした場合、ほとんど同じ結果が得られることを確認しています。