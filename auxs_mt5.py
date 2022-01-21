
import MetaTrader5 as mt5
import pandas as pd

def init_mt5(ativo):
  if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
  mt5.symbol_select(ativo,True)

def get_data(ativo, utc_from, utc_to, timeframe):
  rates = mt5.copy_rates_range(ativo, timeframe, utc_from, utc_to)
  # mt5.shutdown()
  df_raw = pd.DataFrame(rates)
  df_raw['time']=pd.to_datetime(df_raw['time'], unit='s')
  return df_raw

def send_order(symbol, lot=1.0, deviation=10, order_type='buy'):
  # print('Esse é o ativo' + symbol)
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
