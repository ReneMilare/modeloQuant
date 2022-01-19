import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import MetaTrader5 as mt5
import datetime as dt
import auxs
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import math
import pickle

pd.set_option('display.max_rows', 500)

ativo = 'WINFUT'

df = auxs.read_data()

df['Data'] = df.apply(lambda row: row['Data'].split('/')[2]+'-'+row['Data'].split('/')[1]+'-'+row['Data'].split('/')[0], axis=1)

df.time = df.Data + 'T' + df.time

df.time = pd.to_datetime(df.time)

df.drop('Data', inplace=True, axis=1)

df, cols = auxs.add_lags(df, 5)

auxs.set_seeds()

split = int(df.shape[0] * 0.7)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

test.index.name = "Data"
test.reset_index(inplace = True)

test["Data"] = pd.to_datetime(test["time"])


# clf = auxs.create_ensemble_model(train)

feature_selection = sm.GLM(endog=train.target, exog=sm.add_constant(train[cols]), family=sm.families.Binomial()).fit()

df_fs = pd.DataFrame(feature_selection.pvalues, columns=['pvalue'])
df_fs.drop('const', axis=0, inplace=True)

# clf = auxs.logistic_regression_model(train)
# clf = auxs.ensemble_sgd_model(train)
clf = auxs.sgd_model(train)

cols = df_fs[df_fs.pvalue < 0.05].index

clf.fit(train[cols], train.target)

pickle.dump(clf, open('./models/' + ativo + '.sav', 'wb'))

clf = pickle.load(open('./models/' + ativo + '.sav', 'rb'))

train['pred'] = clf.predict(train[cols])
test['pred'] = clf.predict(test[cols])

plt.figure()
print('Matrix do teste')
plot_confusion_matrix(clf, train[cols], train.target)

plt.figure()
print('Matrix do treino')
plot_confusion_matrix(clf, test[cols], test.target)

# test.pred = test.pred.shift()
# test.dropna(inplace=True)

venda = -test.retorno
compra = test.retorno

# test['lag_5'] = test.close.shift(5)
# test.dropna(inplace=True)

test['pode_operar'] = test.apply(lambda row: 
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

regra_inclinacao = 'inclinacao_close'
inclinacao = 40

test['trade'] = np.where(
    (test.pred == 0) & 
    (test.mme_9 < test.mma_20) &
    (test.close < test.mme_9) &
    (test.close < test.mma_20) &
    (test.pode_operar) &
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1) &
        (test.mme_9 > test.mma_20) &
        (test.close > test.mme_9) &
        (test.close > test.mma_20) &
        (test.pode_operar) &
        (test[regra_inclinacao] > inclinacao), compra, 0)
)


test['trade2'] = np.where(
    (test.pred == 0)  &
    (test.pode_operar)&
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1)  &
        (test.pode_operar)&
        (test[regra_inclinacao] > inclinacao), compra, 0)
)

test['trade2_'] = np.where(
    (test.pred == 0)  &
    (test.pode_operar), venda,
    np.where(
        (test.pred == 1)  &
        (test.pode_operar), compra, 0)
)

test['trade3'] = np.where(
    (test.pred == 0) & 
    (test.mme_9 < test.mma_20) &
    (test.pode_operar)&
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1) &
        (test.mme_9 > test.mma_20) &
        (test.pode_operar)&
        (test[regra_inclinacao] > inclinacao), compra, 0)
)

test['trade4'] = np.where(
    (test.pred == 0) & 
    (test.mme_9 < test.mma_20) &
    (test.mma_20 < test.mma_20.shift()) &
    (test.pode_operar)&
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1) &
        (test.mme_9 > test.mma_20) &
        (test.mma_20 > test.mma_20.shift()) &
        (test.pode_operar)&
        (test[regra_inclinacao] > inclinacao), compra, 0)
)

test['trade5'] = np.where(
    (test.pred == 0) & 
    (test.mme_9 < test.mma_20) &
    (test.close < test.mme_9) &
    (test.close < test.mma_20) &
    (test.mma_20 < test.mma_20.shift()) &
    (test.pode_operar)&
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1) &
        (test.mme_9 > test.mma_20) &
        (test.close > test.mme_9) &
        (test.close > test.mma_20) &
        (test.mma_20 > test.mma_20.shift()) &
        (test.pode_operar)&
        (test[regra_inclinacao] > inclinacao), compra, 0)
)

test['trade6'] = np.where(
    (test.pred == 0) & 
    (test.mme_9 < test.mma_20) &
    (test.close < test.mme_9) &
    (test.close < test.mma_20) &
    (test.mma_20 < test.mma_20.shift()) &
    (test.pode_operar)&
    (test[regra_inclinacao] < -inclinacao), venda,
    np.where(
        (test.pred == 1) &
        (test.mme_9 > test.mma_20) &
        (test.close > test.mme_9) &
        (test.close > test.mma_20) &
        (test.mma_20 > test.mma_20.shift()) &
        (test.pode_operar)&
        (test[regra_inclinacao] > inclinacao), compra, 0)
)

# test['coloracao'] = np.where(
#     (test.pred == 0) & 
#     (test.mme_9 < test.mma_20), 'VENDE',
#     np.where(
#         (test.pred == 1) &
#         (test.mme_9 > test.mma_20), 'COMPRA', '--'
#     )
# )

test['lucro'] = test.trade.cumsum()*100
test['lucro2'] = test.trade2.cumsum()*100
test['lucro2_'] = test.trade2_.cumsum()*100
test['lucro3'] = test.trade3.cumsum()*100
test['lucro4'] = test.trade4.cumsum()*100
test['lucro5'] = test.trade5.cumsum()*100
test['lucro6'] = test.trade6.cumsum()*100

fig_size = (15,12)

plt.figure(figsize=fig_size)
plt.title('Trade 1')
plt.plot(test.time, test.lucro)

plt.figure(figsize=fig_size)
plt.title('Trade 2')
plt.plot(test.time, test.lucro2)
plt.figure(figsize=fig_size)
plt.title('Trade 2_')
plt.plot(test.time, test.lucro2_)

plt.figure(figsize=fig_size)
plt.title('Trade 3')
plt.plot(test.time, test.lucro3)

plt.figure(figsize=fig_size)
plt.title('Trade 4')
plt.plot(test.time, test.lucro4)

plt.figure(figsize=fig_size)
plt.title('Trade 5')
plt.plot(test.time, test.lucro5)

plt.figure(figsize=fig_size)
plt.title('Trade 6')
plt.plot(test.time, test.lucro6)

plot_roc_curve(clf, test[cols], test.target)

plt.show()

# daqui para baixo é do Leandro

trade = 'trade2'

test["train_test"] = np.where(test["Data"] > train.time[len(train)-1], 1, -1)

base_agregada = test.resample("M", on = "Data").sum()

base_agregada.loc[: , "Retorno_Modelo_Acumulado"] = base_agregada[trade].cumsum()*100

summary = test.copy()
summary["Data"] = pd.to_datetime(summary["time"], format = "%Y-%m")
summary = summary.groupby([summary["Data"].dt.year]).agg({trade: sum})
summary.index = summary.index.set_names(["Ano"])

summary_mes = test.copy()
summary_mes["Data"] = pd.to_datetime(summary_mes["time"], format = "%Y-%m")
summary_mes = summary_mes.groupby([summary_mes["Data"].dt.year, summary_mes["Data"].dt.month]).agg({trade: sum})
summary_mes.index = summary_mes.index.set_names(["Ano", "Mes"])

summary_days = test.copy()
summary_days["Data"] = pd.to_datetime(summary_days["time"], format = "%Y-%m")
summary_days = summary_days.groupby([summary_days["Data"].dt.year, summary_days["Data"].dt.month, summary_days["Data"].dt.day]).agg({trade: sum})
summary_days.index = summary_days.index.set_names(['Ano','Mes',"Dias"])

summary_hour = test.copy()
summary_hour["Data"] = pd.to_datetime(summary_hour["time"], format = "%Y-%m")
summary_hour = summary_hour.groupby([summary_hour["Data"].dt.year, summary_hour["Data"].dt.month, summary_hour["Data"].dt.day, summary_hour['Data'].dt.hour]).agg({trade: sum})
summary_hour.index = summary_hour.index.set_names(['Ano','Mes',"Dias", 'Horas'])

print()
print('Acc - treino')
print(accuracy_score(train.target, train.pred))

print()
print('Acc - teste')
print(accuracy_score(test.target, test.pred))

print()
print('Report')
print(classification_report(test.target, test.pred))

print("---------------------------------------------------")
print("Pior trade:              {} %".format(round(test[trade].min(), 5)*100))
print("Melhor trade:            {} %".format(round(test[trade].max(), 5)*100))
print("Média trade:             {} %".format(round(test[trade].mean(), 5)*100))
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
print("# Trades negativos:      {}".format((test[trade] < 0).sum()))
print("# Trades positivos:      {}".format((test[trade] > 0).sum()))
print("# Total de trades:       {}".format((test[trade] > 0).sum() + (test[trade] < 0).sum()))
print("# Taxa de acertos:       {} %".format(((test[trade] > 0).sum()/((test[trade] > 0).sum() + (test[trade] < 0).sum()))*100))
print("---------------------------------")
print("# Total(simples):        {} %".format(round(summary_mes[trade].sum(), 5)*100))