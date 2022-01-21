import MetaTrader5 as mt5
import datetime as dt
import pandas as pd
import time
import pickle
import numpy as np
from termcolor import colored
import math
import warnings
import auxs_mt5
import pre_processing
warnings.filterwarnings("ignore")

ativo = 'WDOG22'
modelo = 'WDO@'

auxs_mt5.init_mt5(
    ativo = ativo
)

pre_processing.set_seeds()

while True:
    utc_from = dt.datetime.now() - dt.timedelta(10)
    utc_to = dt.datetime.now()

    df_raw = auxs_mt5.get_data(
        ativo     = ativo, 
        utc_from  = utc_from, 
        utc_to    = utc_to,
        timeframe = mt5.TIMEFRAME_M5
    )

    train, test, cols, df = pre_processing.add_lags(
        data       = df_raw, 
        lags       = 5,
        window     = 20,
        fit_model  = False, 
        ativo      = ativo,
        train_size = 0.7
    )

    clf = pickle.load(open('./models/' + modelo + '.sav', 'rb'))

    cols = pre_processing.get_features(
        ativo = modelo
    )

    df['pred'] = clf.predict(df[cols])

    df_params = pd.read_csv('./features/params_'+ modelo)

    regra_inclinacao = 'linear_angle'
    inclinacao = df_params['mean_angle'].values[0] + df_params['std_angle'].values[0]

    df['trade_print'] = np.where(
      (df.pred == 0)  &
      (df[regra_inclinacao] < -inclinacao), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
      np.where(
          (df.pred == 1)  &
          (df[regra_inclinacao] > inclinacao), colored('######### COMPRA ##########', 'white', 'on_green', attrs=['bold']), colored('--','white','on_cyan'))
    )

    df['trade'] = np.where(
      (df.pred == 0)  &
      (df[regra_inclinacao] < -inclinacao), -1,
      np.where(
          (df.pred == 1)  &
          (df[regra_inclinacao] > inclinacao), 1, 0)
    )

    candle = -2
    lot = 1.0
    deviation = 5

    if df.trade[df.index[candle]] == -1 and mt5.positions_get(symbol=ativo) == ():
        retult = auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='sell'
        )
    elif df.trade[df.index[candle]] == 1 and mt5.positions_get(symbol=ativo) == ():
        result = auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='buy'
        )
    elif df.trade[df.index[candle]] == 0 and mt5.positions_get(symbol=ativo) != ():
        auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='close'
        )

        time.sleep(5)
    elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 0 and df.trade[df.index[candle]] == -1:
        auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='close'
        )
        auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='sell'
        )
    elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 1 and df.trade[df.index[candle]] == 1:
        auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='close'
        )
        auxs_mt5.send_order(
            symbol=ativo,
            lot = lot,
            deviation=deviation,
            order_type='buy'
        )
    

    print(df.trade_print[df.index[-2]])
    print(df.trade_print[df.index[-1]])
    print(df.pred[df.index[-1]])
    print(df.linear_angle[df.index[-1]])
    print(inclinacao)
    print()

