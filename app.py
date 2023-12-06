import streamlit as st
import pandas as pd
import requests
import numpy
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import RandomForestClassifier

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class df(BaseModel):
    RI: float
    Na: float
    Mg: float
    Al: float
    Si: float
    K: float
    Ca: float
    Ba: float
    Fe: float

# モデルの構築
def build_model():
    df = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\train_glass.tsv', index_col=0)
    features = pd.DataFrame(df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], columns=df.columns.tolist()[:-1])
    target = df['Type']

    model = RandomForestClassifier()
    model.fit(features, target)

    return model

# 学習済みのモデルの読み込み
model = build_model()

# /predict エンドポイントの処理
@app.get('/')
def index():
    return {"Glass": 'df_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/predict')
def make_predictions(features: df):
    return {'prediction': str(model.predict([[features.RI, features.Na, features.Mg, features.Al, features.Si, features.K, features.Ca, features.Ba, features.Fe]])[0])}

# Streamlit アプリケーション
if st.button("Run Streamlit App"):
    st.write("The app is running!")

st.title('ガラス工房へようこそ！')

image = Image.open('image.jpg')
st.image(image,use_column_width=True)

st.header('本日はあなたのオリジナルガラスを制作してみましょう')
st.header('')
st.write('私たちの身の回りには多くのガラス製品があります。家の中、車、電車、建物・・')
st.header('')
st.write('まずは、各酸化物含有量からあなたが作ろうとしているガラスの種類を調べます。')
st.write('左の枠にある項目について各値を入力して、一番下の「できあがり」をクリックしてください。')

st.sidebar.header('値を設定してください')
RI = st.sidebar.slider('屈折率', min_value=1.50, max_value=1.55, step=0.01)
Na = st.sidebar.slider('ナトリウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Mg = st.sidebar.slider('マグネシウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Al = st.sidebar.slider('アルミニウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Si = st.sidebar.slider('シリコン (g)', min_value=0.0, max_value=100.0, step=1.0)
K = st.sidebar.slider('カリウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Ca = st.sidebar.slider('カルシウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Ba = st.sidebar.slider('バリウム (g)', min_value=0.0, max_value=100.0, step=1.0)
Fe = st.sidebar.slider('鉄 (g)', min_value=0.0, max_value=100.0, step=1.0)

df = {
    "RI": RI,
    "Na": Na,
    "Mg": Mg,
    "Al": Al,
    "Si": Si,
    "K": K,
    "Ca": Ca,
    "Ba": Ba,
    "Fe": Fe
}

targets = ['加工して使用する建築用ガラス', '未加工で使用する建築用ガラス', '車両用ガラス', 'コンテナ用ガラス', '食器用ガラス', 'ヘッドライト用ガラス', '照明用ガラス', '風鈴用ガラス']

if st.sidebar.button("できあがり"):
    # 入力された説明変数の表示
    st.write('## 入力値')
    glass_df = pd.DataFrame(df, index=["原料一覧"])
    st.write(glass_df)

    # 予測の実行
    response = requests.post("https://glassapp-kh9owc32nt34xnlifhbkgd.streamlit.app/predict", json=df)
    print(response.text)
    prediction = response.json()["prediction"]

    # 予測結果の表示
    st.write('## できあがり')
    st.write(prediction)

    # 予測結果の出力
    st.write('## あなたが制作するオリジナルガラス')
    st.write('それでは早速「',str(targets[int(prediction)]),'」をつくりましょう!')

# image2 = Image.open('C:/Users/ce264/Desktop/glass/image2.jpg')
image2 = Image.open('image2.jpg')
st.image(image2,use_column_width=True)