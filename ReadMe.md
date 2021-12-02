数字予測のアプリケーション部分のみ（model作成部分は別にしました）

sklearnのよりもだいぶ精度高くはなった気がする

データセットはMNIST

フレームワークはTensorFlow、Keras

my_model.h5ってのが別のプログラムで作ったモデル

変えたのはapplivation.pyだけ

画像サイズが小さいほうが精度が高いのでcanvasでかいた画像を読み込むときいったん32x32pxサイズにリサイズしてから処理した

参考サイト：

https://masaki-blog.net/dl-keras-mnist

https://ai-coordinator.jp/mnist
