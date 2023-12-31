import streamlit as st
import pandas as pd
import requests
from PIL import Image

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

glass = {
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
    glass_df = pd.DataFrame(glass, index=["原料一覧"])
    st.write(glass_df)

    # 予測の実行
    response = requests.post("https://gapp4.onrender.com/predict", json=glass)
    prediction = response.json()["prediction"]

    # 予測結果の表示
    st.write('## できあがり')
    st.write(prediction)

    # 予測結果の出力
    st.write('## あなたが制作するオリジナルガラス')
    st.write('それでは早速「',str(targets[int(prediction)]),'」をつくりましょう!')

image2 = Image.open('image2.jpg')
st.image(image2,use_column_width=True)

# streamlit run app.py & uvicorn main:app --reload
# http://localhost:8000/predict"