# RikkaLayer-Sample
## リンク
論文っぽいやつ　：　https://qiita.com/ApfelSchorle/items/82512d68a8d87865b3fa
私への連絡　：　https://twitter.com/KyoumeiProject

## 使い方
RikkaLayer.pyの中のRikkaLayerクラスを、Linear層などと同じように使います。
時系列的に関係ないデータを入力するときは、reset()を呼び出してください。
内部の潜在変数が引き継がれてしまうので上手くいきません。
定義時には、入力次元数を渡せばよいです。
forward()実行時のデータの入力形式は、3次元固定で、[バッチ, 時間軸,　入力次元]となっています。
forward()だけではなく、step()も用意されています。
step()の入力形式は、[バッチ,入力次元]となっています。
出力は入力と同じ形です。
注意点としては、出力にも残差結合を適用してありますので、最後にLinear層を追加してください。
例
batch = 10
SequenseLen = 15
dim = 128
model = RikkaLayer(dim)
#一括入力
data = torch.ones((batch, SequenseLen, dim))
model(data)
#一個ずつ入力
stepData = torch.ones((batch, dim))
model.step(stepData)
