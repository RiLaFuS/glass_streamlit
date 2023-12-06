import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

glass = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\train_glass.tsv', index_col=0)
features = glass.drop('Type', axis=1)
target = glass['Type']

#モデル構築
model = RandomForestClassifier()
model.fit(features, target)

# モデルの保存
import pickle
pickle.dump(model, open('models/model_glass', 'wb'))