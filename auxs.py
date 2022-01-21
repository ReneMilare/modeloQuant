import numpy as np
import keras
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

def cw(df):
  c0, c1= np.bincount(df.target)
  w0 = (1 / c0) * (len(df)) / 2
  w1 = (1 / c1) * (len(df)) / 2
  return {0: w0, 1: w1}

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
    verbose=0
  )

  clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
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
