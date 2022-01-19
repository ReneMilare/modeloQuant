import MetaTrader5 as mt5
import datetime as dt
import pandas as pd
import time
import pickle
import auxs
import numpy as np
from termcolor import colored
import math
import warnings
warnings.filterwarnings("ignore")

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

result = ''

while True:
    utc_from = dt.datetime.now() - dt.timedelta(10)
    utc_to = dt.datetime.now()

    modelo = 'WIN@'
    ativo = 'WDOG22'

    rates = mt5.copy_rates_range(ativo, mt5.TIMEFRAME_M5, utc_from, utc_to)
    df_raw = pd.DataFrame(rates)
    df_raw['time']=pd.to_datetime(df_raw['time'], unit='s')

    df, cols = auxs.add_lags(df_raw, 5, fit_model=False)

    auxs.set_seeds()

    # mt5.shutdown()

    clf = pickle.load(open('./models/' + modelo + '.sav', 'rb'))

    # cols = ''

    df['pred'] = clf.predict(df[cols])

    df['inclinacao_reta'] = df.apply(lambda row: math.degrees(np.arctan((row['close'] - row['lag_5'])/5)), axis=1)

    # print(pred[-1])

    # df['trade'] = np.where(
    # (df.pred == 0) & 
    # (df.mme_9 < df.mma_20), colored('******* VENDA *******', 'white', 'on_red', attrs=['bold']),
    # np.where(
    #     (df.pred == 1) &
    #     (df.mme_9 > df.mma_20), colored('####### COMPRA ########', 'white', 'on_green', attrs=['bold']), colored('---','white','on_cyan'))
    # )
    
    df['trade2'] = np.where(
        (df.pred == 0) & 
        (df.mme_9 < df.mma_20) &
        (df.mma_20 < df.mma_20.shift()) &
        (df.inclinacao_reta < -45), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
        np.where(
            (df.pred == 1) &
            (df.mme_9 > df.mma_20) &
            (df.mma_20 > df.mma_20.shift()) &
            (df.inclinacao_reta > 45), colored('######### COMPRA ##########', 'white', 'on_green', attrs=['bold']), colored('--','white','on_cyan')
        )
    )

    df['trade3'] = np.where(
        (df.pred == 0) & 
        (df.mme_9 < df.mma_20) &
        (df.close < df.mme_9) &
        (df.close < df.mma_20) &
        (df.inclinacao_reta < -45), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
        np.where(
            (df.pred == 1) &
            (df.mme_9 > df.mma_20) &
            (df.close > df.mme_9) &
            (df.close > df.mma_20) &
            (df.inclinacao_reta > 45),  colored('######### COMPRA ##########', 'white', 'on_green', attrs=['bold']), colored('--','white','on_cyan')
        )
    )

    regra_inclinacao = 'inclinacao_close'
    inclinacao = 30

    df['trade4'] = np.where(
        (df.pred == 0) &
        (df[regra_inclinacao] < -inclinacao), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
        np.where(
            (df.pred == 1) &
            (df[regra_inclinacao] > inclinacao), colored('######### COMPRA ##########', 'white', 'on_green', attrs=['bold']), colored('--','white','on_cyan')
        )
    )
    
    df['trade'] = np.where(
        (df.pred == 0) &
        (df[regra_inclinacao] < -inclinacao), -1,
        np.where(
            (df.pred == 1) &
            (df[regra_inclinacao] > inclinacao), 1, 0
        )
    )

    candle = -2

    if df.trade[df.index[candle]] == -1 and mt5.positions_get(symbol=ativo) == ():
        retult = auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='sell'
        )
    elif df.trade[df.index[candle]] == 1 and mt5.positions_get(symbol=ativo) == ():
        result = auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='buy'
        )
    elif df.trade[df.index[candle]] == 0 and mt5.positions_get(symbol=ativo) != ():
        auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='close'
        )

        time.sleep(5)

    elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 0 and df.trade[df.index[candle]] == -1:
        auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='close'
        )
        auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='sell'
        )
    elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 1 and df.trade[df.index[candle]] == 1:
        auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='close'
        )
        auxs.send_order(
            symbol=ativo,
            lot = 1.0,
            deviation=10,
            order_type='buy'
        )
    # elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 0 and df.close[df.index[-1]] > df.high[df.index[-2]]:
    #     auxs.send_order(
    #         symbol=ativo,
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='close'
    #     )
    #     time.sleep(60)
    # elif mt5.positions_get(symbol=ativo) != () and mt5.positions_get(symbol=ativo)[0][5] == 1 and df.close[df.index[-1]] < df.low[df.index[-2]]:
    #     auxs.send_order(
    #         symbol=ativo,
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='close'
    #     )
    #     time.sleep(60)
    

    print(df.trade4[df.index[-2]])

    print(df.trade4[df.index[-1]])
    print()

    # time.sleep(2)
