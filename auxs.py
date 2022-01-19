import numpy as np
import talib
import random
import tensorflow as tf
import keras
import itertools
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.multiclass import OneVsOneClassifier
import MetaTrader5 as mt5
import math

def add_lags(data, lags, window=20, fit_model = True):
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

  for p in to_pdcut:
    df[p] = pd.cut(df[p], 5, labels=False)

  df['lag_5'] = df.close.shift(5)
  df.dropna(inplace=True)
  
  for f in features:
    for lag in range(1, lags + 1):
      col = f'{f}_lag_{lag}'
      df[col] = df[f].shift(lag)
      cols.append(col)
  
  df['inclinacao_close'] = df.apply(lambda row: math.degrees(np.arctan((row['close'] - row['lag_5'])/5)), axis=1)

  df['inclinacao_mma20'] = df.apply(lambda row: math.degrees(np.arctan((row['mma20'] - row['lag_5'])/5)), axis=1)

  df['inclinacao_mme9'] = df.apply(lambda row: math.degrees(np.arctan((row['mme9'] - row['lag_5'])/5)), axis=1)

  df.dropna(inplace=True)

  if fit_model:
    df['target'] = np.where(df.retorno > 0, 1, 0)
    df.target = df.target.shift(-1)
  
  df.dropna(inplace=True)
  return df, cols

def cw(df):
  c0, c1= np.bincount(df.target)
  w0 = (1 / c0) * (len(df)) / 2
  w1 = (1 / c1) * (len(df)) / 2
  return {0: w0, 1: w1}

# def cw(df):
#   c0, c1, c2= np.bincount(df.target)
#   w0 = (1 / c0) * (len(df)) / 3
#   w1 = (1 / c1) * (len(df)) / 3
#   w2 = (1 / c2) * (len(df)) / 3
#   return {0: w0, 1: w1, 2:w2}

def set_seeds(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

def create_model(hl=3, hu=200, cols=1, optimizer=Adam(learning_rate=0.001)):
  model = keras.models.Sequential()
  model.add(
    keras.layers.Dense(hu, input_dim=len(cols), activation='relu')
  )
  for _ in range(hl):
    model.add(
      keras.layers.Dense(hu, activation='relu')
    )
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
      loss='binary_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy']
    )
  return model

def create_ensemble_model(df):

  sgd = SGDClassifier(
    max_iter=5000, 
    tol=1e-3, 
    class_weight=cw(df),
    validation_fraction=0.2,
    n_jobs=-1,
    shuffle=False,
    verbose=1
  )

  random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_split=50,
    n_jobs=-1,
    random_state=42,
    class_weight=cw(df),
    max_leaf_nodes=5,
    verbose=1
  )

  logistic_regression = LogisticRegression(
    class_weight=cw(df),
    random_state=42,
    max_iter=500,
    multi_class='ovr',
    n_jobs=-1,
    verbose=1
  )

  bagg_sgd = BaggingClassifier(
    sgd,
    n_estimators=100,
    bootstrap=True,
    max_samples=0.80,
    n_jobs=-1,
    random_state=42,
    verbose=1
  )

  bagg_random_forest = BaggingClassifier(
    random_forest,
    n_estimators=100,
    bootstrap=True,
    max_samples=0.80,
    n_jobs=-1,
    random_state=42,
    verbose=1
  )

  bagg_logistic_regression = BaggingClassifier(
    logistic_regression,
    n_estimators=100,
    bootstrap=True,
    max_samples=0.80,
    n_jobs=-1,
    random_state=42,
    verbose=1
  )

  clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    StackingClassifier(
      estimators=[
        ('Bagg SGD', bagg_sgd),
        ('Bagg Random Forest', bagg_random_forest),
        ('Bagg Logistic regression', bagg_logistic_regression),
        # ('SGD', sgd),
        # ('Random Forest', random_forest),
        # ('Logistic regression', logistic_regression)
      ],
      n_jobs=-1,
      cv=3,
      passthrough=True,
      verbose=1,
      final_estimator=logistic_regression
    )
  )
  return clf

def sgd_model(df):
  sgd = SGDClassifier(
    max_iter=5000, 
    tol=1e-3, 
    class_weight=cw(df),
    validation_fraction=0.2,
    n_jobs=-1,
    shuffle=False,
    verbose=1
  )

  clf = make_pipeline(
    # StandardScaler(),
    # PCA(n_components=0.99),
    sgd
  )
  return clf

def ensemble_sgd_model(df):
  sgd = SGDClassifier(
    max_iter=5000, 
    tol=1e-3, 
    class_weight=cw(df),
    validation_fraction=0.2,
    n_jobs=-1,
    shuffle=False,
    verbose=1
  )

  bagg_sgd = BaggingClassifier(
    sgd,
    n_estimators=100,
    bootstrap=True,
    max_samples=0.50,
    n_jobs=-1,
    random_state=42,
    verbose=1
  )

  clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    bagg_sgd
  )
  return clf


def random_forest_model(df):
  random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_split=50,
    n_jobs=-1,
    random_state=42,
    class_weight=cw(df),
    max_leaf_nodes=5,
    verbose=1
  )

  clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    random_forest
  )
  return clf

def logistic_regression_model(df):
  logistic_regression = LogisticRegression(
    class_weight=cw(df),
    random_state=42,
    max_iter=500,
    n_jobs=-1,
    verbose=1
  )

  clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    OneVsOneClassifier(logistic_regression, n_jobs=-1)
  )
  return clf

# Envia ordem para o Metatrader
def send_order(symbol = 'WING22', lot=1.0, deviation=10, order_type='buy'):
  symbol_info = mt5.symbol_info(symbol)
  if symbol_info is None:
      print(symbol, "not found, can not call order_check()")
      mt5.shutdown()
      quit()
  
  # se o símbolo não estiver disponível no MarketWatch, adicionamo-lo
  if not symbol_info.visible:
      print(symbol, "is not visible, trying to switch on")
      if not mt5.symbol_select(symbol,True):
          print("symbol_select({}}) failed, exit",symbol)
          mt5.shutdown()
          quit()
  
  positions=mt5.positions_get(symbol=symbol)
  # point = mt5.symbol_info(symbol).point
  # print(positions== ())

  if order_type == 'buy':
    price = mt5.symbol_info_tick(symbol).ask
    print('compra -> ' + str(price))
    request = {
      "action": mt5.TRADE_ACTION_DEAL,
      "symbol": symbol,
      "volume": lot,
      "type": mt5.ORDER_TYPE_BUY,
      "price": price,
      "deviation": deviation,
      "magic": 42,
      "comment": "python script open",
      "type_time": mt5.ORDER_TIME_GTC,
      "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result=mt5.order_send(request)
    if result is None:
        print("order_send() failed, error code =",mt5.last_error())
    if result.retcode != mt5.TRADE_RETCODE_DONE:
      print("2. order_send failed, retcode={}".format(result.retcode))
    return result
  elif order_type == 'sell':
    price=mt5.symbol_info_tick(symbol).bid
    print('venda -> ' + str(price))
    request = {
      "action": mt5.TRADE_ACTION_DEAL,
      "symbol": symbol,
      "volume": lot,
      "type": mt5.ORDER_TYPE_SELL,
      "price": price,
      "deviation": deviation,
      "magic": 42,
      "comment": "python script open",
      "type_time": mt5.ORDER_TIME_GTC,
      "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result=mt5.order_send(request)
    if result is None:
        print("order_send() failed, error code =",mt5.last_error())
    if result.retcode != mt5.TRADE_RETCODE_DONE:
      print("2. order_send failed, retcode={}".format(result.retcode))
    return result
  elif positions != () and positions[0][5] == 0:
    price=mt5.symbol_info_tick(symbol).bid
    request={
      "action": mt5.TRADE_ACTION_DEAL,
      "symbol": symbol,
      "volume": lot,
      "type": mt5.ORDER_TYPE_SELL,
      "position": positions[0][0],
      "price": price,
      "deviation": deviation,
      "magic": 42,
      "comment": "python script close",
      "type_time": mt5.ORDER_TIME_GTC,
      "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result=mt5.order_send(request)
    result = ''
    return result
  elif positions != () and positions[0][5] == 1:
    price=mt5.symbol_info_tick(symbol).ask
    request={
      "action": mt5.TRADE_ACTION_DEAL,
      "symbol": symbol,
      "volume": lot,
      "type": mt5.ORDER_TYPE_BUY,
      "position": positions[0][0],
      "price": price,
      "deviation": deviation,
      "magic": 42,
      "comment": "python script close",
      "type_time": mt5.ORDER_TIME_GTC,
      "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result=mt5.order_send(request)
    result =''
    return result

def read_data():
  df = pd.read_csv('./data/WIN.csv', sep=';')

  return df