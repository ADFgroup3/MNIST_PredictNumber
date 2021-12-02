from flask import Flask,render_template,request, jsonify
import base64
from io import BytesIO

from PIL import Image, ImageOps
import PIL.Image
#機械学習で使うモジュール
import sklearn.datasets
import sklearn.svm
import numpy as np
from keras.models import load_model

app = Flask(__name__)

model_file = 'my_model.h5'
model = load_model(model_file)    

#画像ファイルを数値リストに変換する
def imageToData(filename):
    im = Image.open(filename)
    # 画像ファイル変換
    im = im.resize((28, 28)) # サイズ調整
    im = im.convert('L')     # 白黒変換
    im = ImageOps.invert(im) # 値反転

    # 画像データから検証データ作成
    x = np.array(im)        # NumPy配列に
    x = x.reshape(1, 784)   # 28*28 -> 784
    x = x.astype('float32') # intからfloatに
    x /= 255                # 0-1の範囲のfloatに
    return x

#数字を予測する
def predictDigits(data):
    # 予測
    predict_x=model.predict(data) 
    classes_x=np.argmax(predict_x,axis=1)
    predict_y = np.argmax(model.predict(data), axis=-1)
    return predict_y


@app.route("/", methods=["GET", "POST"])
def main_page():
    #GET処理（何もしない）
    if request.method == 'GET':
        text = "ここに結果が出力されます"
        return render_template("page.html",text=text)

    #POST処理（画像取り込み→処理）
    elif request.method == 'POST':
        #画像データをフォーム内の隠し要素から取得
        img_base64 = request.form['img']
        #base64型式の画像データをデコードして開く
        image = Image.open(BytesIO(base64.b64decode(img_base64)))

        #画像を32x32にする（これしないと精度がた落ちる）
        image = image.resize((32,32))

        #開いた画像データをimagesフォルダーにimage.pngとして保存
        image.save('images/image.png', 'PNG')
        #画像ファイルを数値リストに変換する
        data = imageToData('images/image.png')
        #数字を予測する
        text = predictDigits(data)
        return render_template("page.html",text=text)
## 実行
if __name__ == "__main__":
    app.run(debug=True)