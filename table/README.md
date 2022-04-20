
## ひな形：テーブルデータ分析


1. 最低限のクレンジング
2. プロファイリングとGBT系での簡易モデル化によって重要特徴を絞り込む
3. 変数削減した後に、Poly, Categorical Encodingなどで変数を追加
4. 2へ戻る, このときプロジェクトを複製する(manage.py restartproject projname)


## Best practice

- 目的変数ごとにディレクトリを分ける

- 1iterごとにディレクトリを分ける
  つまり複数回同じコマンドを使うことを想定しいない