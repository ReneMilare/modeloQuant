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

    # ativo = 'WIN@'
    ativo = 'WING22'

    rates = mt5.copy_rates_range(ativo, mt5.TIMEFRAME_M1, utc_from, utc_to)
    df_raw = pd.DataFrame(rates)
    df_raw['time']=pd.to_datetime(df_raw['time'], unit='s')

    df, cols = auxs.add_lags(df_raw, 5, fit_model=False)

    auxs.set_seeds()

    # mt5.shutdown()

    clf = pickle.load(open('./models/' + ativo + '.sav', 'rb'))

    df['pred'] = clf.predict(df[cols])

    df['inclinacao_reta'] = df.apply(lambda row: math.degrees(np.arctan((row['close'] - row['lag_5'])/5)), axis=1)

    # print(pred[-1])

    # df['trade'] = np.where(
    # (df.pred == 0) & 
    # (df.mean_9 < df.mean_20), colored('******* VENDA *******', 'white', 'on_red', attrs=['bold']),
    # np.where(
    #     (df.pred == 1) &
    #     (df.mean_9 > df.mean_20), colored('####### COMPRA ########', 'white', 'on_green', attrs=['bold']), colored('---','white','on_cyan'))
    # )
    
    df['trade2'] = np.where(
        (df.pred == 0) & 
        (df.mean_9 < df.mean_20) &
        (df.mean_20 < df.mean_20.shift()) &
        (df.inclinacao_reta < -45), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
        np.where(
            (df.pred == 1) &
            (df.mean_9 > df.mean_20) &
            (df.mean_20 > df.mean_20.shift()) &
            (df.inclinacao_reta > 45), colored('######### COMPRA ##########', 'white', 'on_green', attrs=['bold']), colored('--','white','on_cyan')
        )
    )

    df['trade3'] = np.where(
        (df.pred == 0) & 
        (df.mean_9 < df.mean_20) &
        (df.close < df.mean_9) &
        (df.close < df.mean_20) &
        (df.inclinacao_reta < -45), colored('********* VENDA *********', 'white', 'on_red', attrs=['bold']),
        np.where(
            (df.pred == 1) &
            (df.mean_9 > df.mean_20) &
            (df.close > df.mean_9) &
            (df.close > df.mean_20) &
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

    # if df.trade[df.index[-2]] == -1 and mt5.positions_get(symbol='WING22') == ():
    #     retult = auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='sell'
    #     )
    # elif df.trade[df.index[-2]] == 1 and mt5.positions_get(symbol='WING22') == ():
    #     result = auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='buy'
    #     )
    # elif df.trade[df.index[-2]] == 0 and mt5.positions_get(symbol='WING22') != ():
    #     auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='close'
    #     )
    # elif mt5.positions_get(symbol='WING22') != () and mt5.positions_get(symbol='WING22')[0][5] == 0 and df.trade[df.index[-2]] == -1:
    #     auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='close'
    #     )
    #     auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='sell'
    #     )
    # elif mt5.positions_get(symbol='WING22') != () and mt5.positions_get(symbol='WING22')[0][5] == 1 and df.trade[df.index[-2]] == 1:
    #     auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='close'
    #     )
    #     auxs.send_order(
    #         symbol='WING22',
    #         lot = 1.0,
    #         deviation=10,
    #         order_type='buy'
    #     )

    # print(result)
    # print(result[0])
    # if mt5.positions_get(symbol='WING22') != ():
    #     print(mt5.positions_get(symbol='WING22'))
    #     print(mt5.positions_get(symbol='WING22')[0][7])
    # print(df.trade4[df.index[-2]])
    print(df.trade4[df.index[-2]])
    # print(df.time[df.index[-2]])
    # print(df.close[df.index[-2]])
    # print(df.trade[df.index[-2]])

    print(df.trade4[df.index[-1]])
    # print(df.time[df.index[-1]])
    # print(df.close[df.index[-1]]) 
    # print(df.trade[df.index[-1]])
    # print(result)
    # print(mt5.positions_get(symbol='WING22')[0])
    # print()
    # print(mt5.positions_get(symbol='WING22')[0][5])
    # print(mt5.positions_get(symbol='WING22')['type'])
    # print(mt5.symbol_info("WING22"))
    # print(df.close[df.index[-1]])
    # print(df.index[-1]-1)
    # print(df.time[df.index[-2]])
    # print(df.time[df.index[-1]])
    # print(df.time.tail(3))
    # print(df_raw.close.tail(3))
    # print(df.close.tail(3))
    # print(df.trade2.tail(3))
    print()

    time.sleep(2)
