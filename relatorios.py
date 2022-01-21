import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

def get_plots(df, clf, cols):

  fig_size = (15,12)

  plt.figure(figsize=fig_size)
  plt.title('Trade 1')
  plt.plot(df.time, df.lucro)

  plt.figure(figsize=fig_size)
  plt.title('Trade 2')
  plt.plot(df.time, df.lucro2)
  plt.figure(figsize=fig_size)
  plt.title('Trade 2_')
  plt.plot(df.time, df.lucro2_)

  plt.figure(figsize=fig_size)
  plt.title('Trade 3')
  plt.plot(df.time, df.lucro3)

  plt.figure(figsize=fig_size)
  plt.title('Trade 4')
  plt.plot(df.time, df.lucro4)

  plt.figure(figsize=fig_size)
  plt.title('Trade 5')
  plt.plot(df.time, df.lucro5)

  plt.figure(figsize=fig_size)
  plt.title('Trade 6')
  plt.plot(df.time, df.lucro6)

  plot_roc_curve(clf, df[cols], df.target)

  plt.show()

def make_back_test(df, custo, regra_inclinacao, inclinacao):
  
  venda = -df.retorno - custo
  compra = df.retorno - custo

  df['pode_operar'] = df.apply(lambda row: 
      row['Data'] > pd.Timestamp(
          year= row.Data.year, 
          month=row.Data.month,
          day=row.Data.day,
          hour=9,
          minute=5
      ) and
      row['Data'] < pd.Timestamp(
          year= row.Data.year, 
          month=row.Data.month,
          day=row.Data.day,
          hour=17,
          minute=0
      ), axis=1
  )

  df['trade'] = np.where(
      (df.pred == 0) & 
      (df.mme_9 < df.mma_20) &
      (df.close < df.mme_9) &
      (df.close < df.mma_20) &
      (df.pode_operar) &
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1) &
          (df.mme_9 > df.mma_20) &
          (df.close > df.mme_9) &
          (df.close > df.mma_20) &
          (df.pode_operar) &
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )


  df['trade2'] = np.where(
      (df.pred == 0)  &
      (df.pode_operar)&
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1)  &
          (df.pode_operar)&
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )

  df['trade2_'] = np.where(
      (df.pred == 0)  &
      (df.pode_operar), venda,
      np.where(
          (df.pred == 1)  &
          (df.pode_operar), compra, 0)
  )

  df['trade3'] = np.where(
      (df.pred == 0) & 
      (df.mme_9 < df.mma_20) &
      (df.pode_operar)&
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1) &
          (df.mme_9 > df.mma_20) &
          (df.pode_operar)&
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )

  df['trade4'] = np.where(
      (df.pred == 0) & 
      (df.mme_9 < df.mma_20) &
      (df.mma_20 < df.mma_20.shift()) &
      (df.pode_operar)&
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1) &
          (df.mme_9 > df.mma_20) &
          (df.mma_20 > df.mma_20.shift()) &
          (df.pode_operar)&
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )

  df['trade5'] = np.where(
      (df.pred == 0) & 
      (df.mme_9 < df.mma_20) &
      (df.close < df.mme_9) &
      (df.close < df.mma_20) &
      (df.mma_20 < df.mma_20.shift()) &
      (df.pode_operar)&
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1) &
          (df.mme_9 > df.mma_20) &
          (df.close > df.mme_9) &
          (df.close > df.mma_20) &
          (df.mma_20 > df.mma_20.shift()) &
          (df.pode_operar)&
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )

  df['trade6'] = np.where(
      (df.pred == 0) & 
      (df.mme_9 < df.mma_20) &
      (df.close < df.mme_9) &
      (df.close < df.mma_20) &
      (df.mma_20 < df.mma_20.shift()) &
      (df.pode_operar)&
      (df[regra_inclinacao] < -inclinacao), venda,
      np.where(
          (df.pred == 1) &
          (df.mme_9 > df.mma_20) &
          (df.close > df.mme_9) &
          (df.close > df.mma_20) &
          (df.mma_20 > df.mma_20.shift()) &
          (df.pode_operar)&
          (df[regra_inclinacao] > inclinacao), compra, 0)
  )

  df['lucro'] = df.trade.cumsum()*100
  df['lucro2'] = df.trade2.cumsum()*100
  df['lucro2_'] = df.trade2_.cumsum()*100
  df['lucro3'] = df.trade3.cumsum()*100
  df['lucro4'] = df.trade4.cumsum()*100
  df['lucro5'] = df.trade5.cumsum()*100
  df['lucro6'] = df.trade6.cumsum()*100
  return df

def get_relatorio(df, trade):  
  summary = df.copy()
  summary["Data"] = pd.to_datetime(summary["time"], format = "%Y-%m")
  summary = summary.groupby([summary["Data"].dt.year]).agg({trade: sum})
  summary.index = summary.index.set_names(["Ano"])

  summary_mes = df.copy()
  summary_mes["Data"] = pd.to_datetime(summary_mes["time"], format = "%Y-%m")
  summary_mes = summary_mes.groupby([summary_mes["Data"].dt.year, summary_mes["Data"].dt.month]).agg({trade: sum})
  summary_mes.index = summary_mes.index.set_names(["Ano", "Mes"])

  summary_days = df.copy()
  summary_days["Data"] = pd.to_datetime(summary_days["time"], format = "%Y-%m")
  summary_days = summary_days.groupby([summary_days["Data"].dt.year, summary_days["Data"].dt.month, summary_days["Data"].dt.day]).agg({trade: sum})
  summary_days.index = summary_days.index.set_names(['Ano','Mes',"Dias"])

  summary_hour = df.copy()
  summary_hour["Data"] = pd.to_datetime(summary_hour["time"], format = "%Y-%m")
  summary_hour = summary_hour.groupby([summary_hour["Data"].dt.year, summary_hour["Data"].dt.month, summary_hour["Data"].dt.day, summary_hour['Data'].dt.hour]).agg({trade: sum})
  summary_hour.index = summary_hour.index.set_names(['Ano','Mes',"Dias", 'Horas'])

  print()
  print('Report')
  print(classification_report(df.target, df.pred))

  print("---------------------------------------------------")
  print("Pior trade:              {} %".format(round(df[trade].min(), 5)*100))
  print("Melhor trade:            {} %".format(round(df[trade].max(), 5)*100))
  print("Média trade:             {} %".format(round(df[trade].mean(), 5)*100))
  print("---------------------------------------------------")
  print("Pior retorno horário:    {} %".format(round(summary_hour[trade].min(), 5)*100))
  print("Melhor retorno horário:  {} %".format(round(summary_hour[trade].max(), 5)*100))
  print("Média ganho horário:     {} %".format(round(summary_hour[trade].mean(), 5)*100))
  print("---------------------------------------------------")
  print("Pior retorno diário:     {} %".format(round(summary_days[trade].min(), 5)*100))
  print("Melhor retorno diário:   {} %".format(round(summary_days[trade].max(), 5)*100))
  print("Média ganho diário:      {} %".format(round(summary_days[trade].mean(), 5)*100))
  print("---------------------------------------------------")
  print("Pior retorno mensal:     {} %".format(round(summary_mes[trade].min(), 5)*100))
  print("Melhor retorno mensal:   {} %".format(round(summary_mes[trade].max(), 5)*100))
  print("Média ganho mensal:      {} %".format(round(summary_mes[trade].mean(), 5)*100))
  print("---------------------------------")
  print("Pior retorno anual:      {} %".format(round(summary[trade].min(), 5)*100))
  print("Melhor retorno anual:    {} %".format(round(summary[trade].max(), 5)*100))
  print("Média ganho anual:       {} %".format(round(summary[trade].mean(), 5)*100))
  print("---------------------------------")
  print("# Anos negativos:        {}".format((summary[trade] < 0).sum()))
  print("# Anos positivos:        {}".format((summary[trade] > 0).sum()))
  print("---------------------------------")
  print("# Meses negativos:       {}".format((summary_mes[trade] < 0).sum()))
  print("# Meses positivos:       {}".format((summary_mes[trade] > 0).sum()))
  print("---------------------------------")
  print("# Dias negativos:        {}".format((summary_days[trade] < 0).sum()))
  print("# Dias positivos:        {}".format((summary_days[trade] > 0).sum()))
  print("---------------------------------")
  print("# Horas negativos:       {}".format((summary_hour[trade] < 0).sum()))
  print("# Horas positivos:       {}".format((summary_hour[trade] > 0).sum()))
  print("---------------------------------")
  print("# Trades negativos:      {}".format((df[trade] < 0).sum()))
  print("# Trades positivos:      {}".format((df[trade] > 0).sum()))
  print("# Total de trades:       {}".format((df[trade] > 0).sum() + (df[trade] < 0).sum()))
  print("# Taxa de acertos:       {} %".format(((df[trade] > 0).sum()/((df[trade] > 0).sum() + (df[trade] < 0).sum()))*100))
  print("---------------------------------")
  print("# Total(simples):        {} %".format(round(summary_mes[trade].sum(), 5)*100))

def plot_confusion_matrix(clf, df, cols):
  
  print()
  print('Acc')
  print(accuracy_score(df.target, df.pred))

  plt.figure()
  print('Matrix de confusão')
  plot_confusion_matrix(clf, df[cols], df.target)