from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class glass(BaseModel):
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

# トップページ
@app.get('/')
def index():
    return {"Type": 'glass_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/predict')
def make_predictions(features: glass):
    return({'prediction':str(model.predict([[features.RI, features.Na, features.Mg, features.Al, features.Si, features.K, features.Ca, features.Ba, features.Fe]])[0])})