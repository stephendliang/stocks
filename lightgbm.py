# -*- coding: utf-8 -*-

# -- Sheet --


import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.fft import fft, ifft
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import sklearn
import sklearn.model_selection
from sklearn.metrics import log_loss
import time
import warnings

import numpy as np
#import optuna.integration.lightgbm as lgb

#from lightgbm import early_stopping
#from lightgbm import log_evaluation
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from tqdm import tqdm
from scipy.stats import percentileofscore
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

from indicators import *

#from flaml import AutoML
import lightgbm as lgb


def get_df(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='5y')
    return df

fintech_tickers = ['SQ','AFRM','UPST','AFRM','PYPL',
                   'INTU','V','MA','COIN','MELI','SE',
                   'SHOP','FISV','GPN','COIN','MELI','ADYEY']
bluechip_tech_tickers = ['AAPL','AMZN','GOOG','NVDA',
                         'FB','MSFT','TSLA']
cybersecurity_tickers = ['ZS','NET','CRWD','PANW']
ai_tickers = ['FB','GOOG','TSLA','NVDA']

semiconductor_tickers = [
 'ADI','AMAT','AMD','AMKR','ASML','AVGO','AZTA','COHR','ENTG','INTC','IPGP',
 'KLAC','LRCX','LSCC','MCHP','MPWR','MRVL','MU','NVDA','NXPI','ON','POWI','QCOM',
 'QRVO','SLAB','SNPS','SOXX','SWKS','TER','TSM','TXN','UMC','WOLF']
semiconductor_dfs = dict(zip(semiconductor_tickers, [get_df(t) for t in semiconductor_tickers]))


#for key in semiconductor_dfs.keys():
key = 'NVDA'
df = semiconductor_dfs[key]

# long operations
incf3  = max_inc_forward_adj(df.High.values,df.Low.values,3)
incf7  = max_inc_forward_adj(df.High.values,df.Low.values,7)

inct1  = max_inc_forward1(df.High.values,df.Low.values)
inct3  = max_inc_forward(df.High.values,df.Low.values,3)
inct7  = max_inc_forward(df.High.values,df.Low.values,7)

# short operations
dect3  = max_dec_forward(df.High.values, df.Low.values, 3)
dect7  = max_dec_forward(df.High.values, df.Low.values, 7)

# # Tests
params_test = {
    'stoch': 5,
    'stochrsi_rsip': 0,
    'stochrsi_stochp': 0, 

    'rsi': 0,

    'rsi_macross_rsip': 0,
    'rsi_macross_map': 0,

    'rsi_macrossvel_rsip': 0,
    'rsi_macrossvel_map': 0,

    'rsi_macrosspctile_rsip': 0,
    'rsi_macrosspctile_map': 0,

    'gambler0': 0,
    'gambler1': 0,

    'changespctile_meanp': 0,
    'changespctile_lookback': 0,

    #'reversal_days': 1,
    #'reversal_days': 7,

    'powerpctile_p0': 0,
    'powerpctile_p1': 0,
    'powerEBBPPercentile_lookback': 0,
}

def addData(ohlc, title):
    df = dict()
    df['CLOSE_CHANGE']    = ohlc.Close.pct_change()[200:]
    df['OPEN_CHANGE']     = ohlc.Open.pct_change()[200:]
    df['INTRADAY_CHANGE'] = (ohlc.Close / ohlc.Open)[200:]
    #df['BBANDPCT']     = PERCENT_B(df, period=11)
    #df['BBANDPCTILE']  = PERCENTILE_B(df, period=11)
    #df['BBANDWPCTILE'] = BBWIDTHPCTILE(ohlc, period=20)
        
    df['STOCH']        = STOCH(ohlc, period=4)[200:]
    #df['STOCHD']       = STOCHD(df, period=3, stoch_period=3)
    df['STOCHRSI']     = STOCHRSI(ohlc, rsi_period=3, stoch_period=20)[200:]
        
    df['RSI']          = RSI(ohlc=ohlc, period=8)[200:]
    #df['RSIPCTILE']    = RSI(df, )
    #df['RSIPCT']       = RSI(df, )
    #df['RSIMACROSS']   = rsi_cross_ma_pct(df[-400:],d1,d2,d3,'div',mt,sc)
    #df['RSIMACROSS']   = rsi_cross_ma_pct(df[-400:],d1,d2,d3,'div',mt,sc)
        
    #df['CCI']          = CCI(ohlc=df,period=30,constant=20)
    #df['MACROSS']      = MACross(ohlc, d1=8, d2=28)[200:]
    df['BULLBEAR']     = BBP(ohlc, day1=15, day2=30, lookback=100, l=1.25)[200:]
    df['GAMBLER']      = gamblers_fallacy_inter(ohlc, d1=50, d2=100)[200:]

    return pd.DataFrame.from_dict(df)

def calculate_ys(dflist):
    yl = np.zeros(shape=(3000,len(dflist.keys())))

    for idx, key in enumerate(dflist):
        s_df = dflist[key]
        y    = inc

        inct1  = max_inc_forward1(s_df.High.values,df.Low.values)
        dect3  = max_dec_backward(s_df.High.values, df.Low.values, 3)

        yl[(idx<<1)  ] = incf3[3:]
        yl[(idx<<1)+1] = incf3[3:]

    return yl

# # Training

'''

def train(data, target):    
    train_x = data[:-10]
    val_x   = data[-10]
    train_y = target[:-10]
    val_y   = target[-10]

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    #prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
    #accuracy = accuracy_score(val_y, prediction)

    #best_params = model.params
    #print("Best params:", best_params)
    #print("  Accuracy = {}".format(accuracy))
    #print("  Params: ")
    #for key, value in best_params.items():
    #    print("    {}: {}".format(key, value))

    #addData(df, params)
    gbm = lgb.LGBMRegressor(num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=20)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(5)])

def train2(X_train, y_train):
    print(X_train, y_train)
    automl = lightgbm.LGBMRegressor()

    # Specify automl goal and constraint
    automl_settings = {
        "time_budget": 1,  # in seconds
        "metric": 'l2',
        "task": 'regression',
        "log_file_name": "loss.log",
    }

    # Train with labeled input data
    automl.fit(X_train=X_train, y_train=y_train,
               **automl_settings)
    # Predict
    print(automl.predict(X_train))
    # Print the best model
    print(automl.model.estimator)


'''

SEARCH_PARAMS = {'learning_rate': 0.4,
                'max_depth': 15,
                'num_leaves': 32,
                'feature_fraction': 0.8,
                'subsample': 0.2}

FIXED_PARAMS={'objective': 'regression',
             'metric': 'tweedie',
             'is_unbalance':True,
             'bagging_freq':5,
             'boosting':'dart',
             'num_boost_round':300,
             'early_stopping_rounds':30}

def train3(data, target):
    X_train  = data.iloc[:-10]
    X_test   = data.iloc[-10:]
    y_train  = target[:-10]
    print(X_train)
    print(X_train.values)
    print(len(y_train))
    y_test   = target[-10:]
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    params = {'metric':FIXED_PARAMS['metric'],
             'objective':FIXED_PARAMS['objective'],
             **search_params}

    model = lgb.train(params, train_data,
                     valid_sets=[valid_data],
                     num_boost_round=FIXED_PARAMS['num_boost_round'],
                     early_stopping_rounds=FIXED_PARAMS['early_stopping_rounds'],
                     valid_names=['valid'])
    return score



def run():
    df = addData(semiconductor_dfs['NVDA'],'NVDA')
    train3(df, inct1[200:])

run()



