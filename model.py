import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\train_glass.tsv', index_col=0)
df_test = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\test_glass.tsv', index_col=0)

# x = df.drop('Type', axis=1)
features = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], columns=df.columns.tolist()[:-1])
target = df['Type']

#モデル構築
model = RandomForestClassifier()
model.fit(features, target)

# モデルの保存
import pickle
pickle.dump(model, open('models/model_glass', 'wb'))