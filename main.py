from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class Glass(BaseModel):
    RI: float
    Na: float
    Mg: float
    Al: float
    Si: float
    K: float
    Ca: float
    Ba: float
    Fe: float

# 学習済みのモデルの読み込み
model = pickle.load(open('models/model_glass', 'rb'))

# /predict エンドポイントの処理
@app.post("/predict")
async def predict(glass: Glass):
    try:
        # 予測の実行
        prediction = model.predict([[glass.RI, glass.Na, glass.Mg, glass.Al, glass.Si, glass.K, glass.Ca, glass.Ba, glass.Fe]])[0]
        return {"prediction": int(prediction)}  # 予測結果を整数型に変換して返す
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))