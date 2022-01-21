import pandas as pd
import datetime as dt
import auxs
import pickle
import pre_processing
import auxs_mt5
import relatorios
import MetaTrader5 as mt5

pd.set_option('display.max_rows', 500)

ativo = 'WDO@'

auxs_mt5.init_mt5(
    ativo = ativo
)

utc_from = dt.datetime.now() - dt.timedelta(9999)
utc_to = dt.datetime.now()

df_raw = auxs_mt5.get_data(
    ativo     = ativo, 
    utc_from  = utc_from, 
    utc_to    = utc_to,
    timeframe = mt5.TIMEFRAME_M5
)

# df_raw = pre_processing.read_data(
#     dataset = 'WDOFUT_F_0_5min'
# )

train, test, cols, df_retorno = pre_processing.add_lags(
    data       = df_raw, 
    lags       = 5,
    window     = 20,
    fit_model  = True, 
    ativo      = ativo,
    train_size = 0.7
)

pre_processing.set_seeds()

pre_processing.save_features(
    df    = train, 
    cols  = cols, 
    ativo = ativo
)

cols = pre_processing.get_features(
    ativo = ativo
)

clf = auxs.sgd_model(
    df = train
)

clf.fit(train[cols], train.target)
train['pred'] = clf.predict(train[cols])
test['pred'] = clf.predict(test[cols])

pickle.dump(clf, open('./models/' + ativo + '.sav', 'wb'))

clf = pickle.load(open('./models/' + ativo + '.sav', 'rb'))

df_params = pd.read_csv('./features/params_'+ ativo)

regra_inclinacao = 'linear_angle'
inclinacao = df_params['mean_angle'].values[0] + df_params['std_angle'].values[0]

df = relatorios.make_back_test(
    df               = test,
    custo            = 0,
    regra_inclinacao = regra_inclinacao,
    inclinacao       = inclinacao
)

relatorios.get_plots(
    df   = df,
    clf  = clf,
    cols = cols
)

relatorios.get_relatorio(
    df    = df,
    trade = 'trade2'
)