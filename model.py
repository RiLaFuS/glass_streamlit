import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\train_glass.tsv', index_col=0)
df_test = pd.read_table('C:\\Users\\ce264\\Desktop\\glass\\test_glass.tsv', index_col=0)

x = df.drop('Type', axis=1)
features = pd.DataFrame(df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], columns=df.columns.tolist()[:-1])
t = df['Type']

from sklearn.model_selection import train_test_split
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.2, random_state=1)
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=1)
x_train.shape, x_val.shape, x_test.shape

from sklearn.ensemble import RandomForestClassifier

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial, x, t, cv):
  # 1.ハイパーパラメータごとに探索範囲を指定
  max_depth = trial.suggest_int('max_depth', 10000, 50000)
  n_estimators = trial.suggest_int('n_estimators',2,  1000)
  min_samples_split = trial.suggest_int('min_samples_split', 2, 1000)

  # 2.学習に使用するアルゴリズムを指定
  estimator = RandomForestClassifier(
      max_depth = max_depth,
      n_estimators = n_estimators,
      min_samples_split = min_samples_split
  )

  # 3.学習の実行、検証結果の表示
  print('Current_params:', trial.params)
  accuracy = cross_val_score(estimator, x, t, cv=cv).mean()
  return accuracy

# studyオプジェクトの作成（正解率の最大化 direction=maximize）
study = optuna.create_study(sampler=optuna.samplers.TPESampler(0), direction='maximize')

# K値
cv = 5
# 目的関数の最適化
study.optimize(lambda trial: objective(trial, x_train_val, t_train_val, cv), n_trials=10)

# 最適なハイパーパラメータを設定したモデルの定義
best_model = RandomForestClassifier(**study.best_params)
#モデルの学習
best_model.fit(x_train_val,t_train_val)

# モデルの保存
import pickle
pickle.dump(best_model, open('models/model_glass', 'wb'))