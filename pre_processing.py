
import numpy as np
import talib
import random
import tensorflow as tf
import itertools
import pandas as pd
import math
from sklearn.feature_selection import SelectKBest, chi2

def add_lags(data, lags, window=20, fit_model = True, ativo='', train_size=0.7):
  cols = []
  features = []
  to_pdcut = []

  df = data.copy()
  # df['retorno'] = np.log(df.close / df.close.shift())
  df['retorno'] = df.close.pct_change()
  df['min'] = df.close.rolling(window).min()
  df['max'] = df.close.rolling(window).max()
  df['mom'] = df.retorno.rolling(window).mean()
  df['vol'] = df.retorno.rolling(window).std()

  beta_periods = [100, 150, 200]

  for beta in beta_periods:
    combinations = list(itertools.permutations(['open', 'high', 'low', 'close'],2))
    for combination in combinations:
      col = f'beta_{combination[0]}_{combination[1]}_{beta}'
      df[col] = talib.BETA(df[combination[0]], df[combination[1]], timeperiod=beta)
      df[col] = pd.qcut(df[col], 5, labels=False)
      features.append(col)
      cols.append(col)

  df['mma20'] = talib.MA(df.close, timeperiod=20)
  df['mme9'] = talib.EMA(df.close, timeperiod=9)
  df['mma_20'] = talib.MA(df.close, timeperiod=20)
  df['mme_9'] = talib.EMA(df.close, timeperiod=9)

  df['alta_ou_baixa'] = np.where(df.open <= df.close, 1, 0)
  if fit_model:
    df['alvo'] = df.alta_ou_baixa.shift(-1)
  # df['mean_9'] = df.close.rolling(9).mean()
  # df['mean_20'] = df.close.rolling(20).mean()

  df['volatilidade'] = ((df.high - df.low)/df.low) * 100
  df['std_9'] = df.close.rolling(9).std()*100
  df['std_20'] = df.close.rolling(20).std()*100
  df['std_50'] = df.close.rolling(50).std()*100
  df['distancia_mea9'] = ((df.close - df.mme9)/df.mme9) * 100
  df['distancia_mma20'] = ((df.close - df.mma20)/df.mma20) * 100
  df['corpo_candle'] = ((df.close - df.open)/df.open) * 100
  df['distancia_medias'] = ((df.mma20 - df.mme9)/df.mme9) * 100

  # teste de novas variáveis
  df["V_Max"] = df["retorno"].rolling(15).max()
  df["V_Min"] = df["retorno"].rolling(15).min()
  df["I"] = df["retorno"].rolling(15).sum()

  # RSL std5
  df["RSL_std9"] = (df["std_9"]/df["std_9"].rolling(15).mean())-1
  # RSL std10
  df["RSL_std20"] = (df["std_20"]/df["std_20"].rolling(15).mean())-1
  # RSL std15
  df["RSL_std15"] = (df["std_50"]/df["std_50"].rolling(15).mean())-1
  # RSL5 do fechamento
  df["RSL_5"] = (df["close"]/df["close"].rolling(5).mean())-1
  # RSL10 do fechamento
  df["RSL_10"] = (df["close"]/df["close"].rolling(10).mean())-1
  # RSL15 do fechamento
  df["RSL_15"] = (df["close"]/df["close"].rolling(15).mean())-1

  df['linear_reg'] = talib.LINEARREG(df.close, timeperiod=5)
  df['dist_linear_reg'] = ((df.close/df.linear_reg)-1)*100

  features.append('min')
  features.append('max')
  features.append('mom')
  features.append('vol')
  features.append('mma20')
  features.append('mme9')
  features.append('volatilidade')
  features.append('std_9')
  features.append('std_20')
  features.append('std_50')
  features.append('distancia_mea9')
  features.append('distancia_mma20')
  features.append('corpo_candle')
  features.append('distancia_medias')
  features.append('V_Max')
  features.append('V_Min')
  features.append('I')
  features.append('RSL_std9')
  features.append('RSL_std20')
  features.append('RSL_std15')
  features.append('RSL_5')
  features.append('RSL_10')
  features.append('RSL_15')
  features.append('dist_linear_reg')

  to_pdcut.append('min')
  to_pdcut.append('max')
  to_pdcut.append('mom')
  to_pdcut.append('vol')
  to_pdcut.append('mma20')
  to_pdcut.append('mme9')
  to_pdcut.append('volatilidade')
  to_pdcut.append('std_9')
  to_pdcut.append('std_20')
  to_pdcut.append('std_50')
  to_pdcut.append('distancia_mea9')
  to_pdcut.append('distancia_mma20')
  to_pdcut.append('corpo_candle')
  to_pdcut.append('distancia_medias')
  to_pdcut.append('V_Max')
  to_pdcut.append('V_Min')
  to_pdcut.append('I')
  to_pdcut.append('RSL_std9')
  to_pdcut.append('RSL_std20')
  to_pdcut.append('RSL_std15')
  to_pdcut.append('RSL_5')
  to_pdcut.append('RSL_10')
  to_pdcut.append('RSL_15')
  to_pdcut.append('dist_linear_reg')

  for p in to_pdcut:
    df[p] = pd.cut(df[p], 5, labels=False)

  df['lag_5_close'] = df.close.shift(5)
  df['lag_5_mma_20'] = df.mma_20.shift(5)
  df['lag_5_mme_9'] = df.mme_9.shift(5)
  df.dropna(inplace=True)
  
  for f in features:
    for lag in range(1, lags + 1):
      col = f'{f}_lag_{lag}'
      df[col] = df[f].shift(lag)
      cols.append(col)
  
  delta_t = 25

  df['inclinacao_close'] = df.apply(lambda row: math.degrees(np.arctan((row['close'] - row['lag_5_close'])/delta_t)), axis=1)

  df['inclinacao_mma20'] = df.apply(lambda row: math.degrees(np.arctan((row['mma_20'] - row['lag_5_mma_20'])/delta_t)), axis=1)

  df['inclinacao_mme9'] = df.apply(lambda row: math.degrees(np.arctan((row['mme_9'] - row['lag_5_mme_9'])/delta_t)), axis=1)

  df.dropna(inplace=True)

  if fit_model:
    df['target'] = np.where(df.retorno > 0, 1, 0)
    df.target = df.target.shift(-1)

  df['linear_angle'] = talib.LINEARREG_ANGLE(df.close, 5)
  df['linear_slope'] = talib.LINEARREG_SLOPE(df.close, 5)
  
  df.dropna(inplace=True)

  split = int(df.shape[0] * train_size)

  train = df.iloc[:split].copy()
  test = df.iloc[split:].copy()
  
  df_params = pd.DataFrame(data={
    'std_slope' : train.linear_slope.std(),
    'mean_slope' : train.linear_slope.mean(),
    'std_angle' : train.linear_angle.std(),
    'mean_angle' : train.linear_angle.mean()
  }, index=[0])

  df_params.to_csv('./features/params_' + ativo)

  test.index.name = "Data"
  test.reset_index(inplace = True)

  test["Data"] = pd.to_datetime(test["time"])
  return train, test, cols, df

def read_data(dataset):
  df = pd.read_csv('./data/' + dataset + '.csv', sep=';')
  df['Data'] = df.apply(lambda row: row['Data'].split('/')[2]+'-'+row['Data'].split('/')[1]+'-'+row['Data'].split('/')[0], axis=1)

  df.time = df.Data + 'T' + df.time

  df.time = pd.to_datetime(df.time)

  df.drop('Data', inplace=True, axis=1)

  df = df.iloc[::-1]
  return df

def set_seeds(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

def save_features(df, cols, ativo):
  select_features = SelectKBest(chi2, k='all').fit(df[cols], df.target)

  df_features = pd.DataFrame(data={
      'feature': select_features.feature_names_in_,
      'pvalues': select_features.pvalues_
  })
  df_features.to_csv('./features/' + ativo)

def get_features(ativo):
  df_features = pd.read_csv('./features/' + ativo)
  return df_features[df_features.pvalues < 0.05].feature.values
