from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

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
    df_train = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\train_glass.tsv', index_col=0)
    features = pd.DataFrame(df_train[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], columns=df_train.columns.tolist()[:-1])
    target = df_train['Type']

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