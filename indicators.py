# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore


def max_dec_forward(highs, lows, days):
    diff = []
    for i in range(len(highs)-days):
        min_price = min(lows[i:(i+days)])
        diff += [1 - (min_price / highs[i])]
    return diff + [None for _ in range(days)]

def max_inc_forward_adj(highs, lows, days):
    diff = []
    for i in range(len(highs)-days):
        max_price = max(highs[(i+1):(i+days)])
        max_locat = np.argmax(highs[(i+1):(i+days)])
        diff += [((max_price / lows[i]) - 1) / (max_locat + 1)]
    return diff + [None for _ in range(days)]

def max_inc_forward(highs, lows, days):
    diff = []
    for i in range(len(highs)-days):
        max_price = max(highs[(i+1):(i+days)])
        diff += [(max_price / lows[i]) - 1]
    return diff + [None for _ in range(days)]

def max_inc_forward1(highs, lows):
    diff = []
    for i in range(len(highs)-1):
        diff += [(highs[i+1] / lows[i]) - 1]
    return diff + [None]

# # Indicators


# ## Moving Percentile


def moving_percentile(df, days):
    d = [None for _ in range(days)]
    for i in range(days, len(df)):
        d += [percentileofscore(df[(i-days):(i+1)], df[i])]
    return d

# ## MAs


def ZLEMA(ohlc,period = 26,adjust = True):
    lag = int((period - 1) / 2)
    ema = ohlc + (ohlc.diff(lag))
    zlema = ema.ewm(span=period, adjust=adjust).mean()
    return zlema

# ## MA Cross


def MACross(df, d1, d2):
    return (ZLEMA(df,d1) - ZLEMA(df,d2)) / df.Close

# ## TP


def TP(ohlc):
    return pd.Series((ohlc.High + ohlc.Low + ohlc.Close) / 3, name="TP")

# ## RSI


def RSI(ohlc, period=14, column="Close", scale='linear'):
    if column == "Close":
        delta = ohlc[column].diff()
    else:
        delta = (ohlc['High'] - ohlc['Low']) / 2
    
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    if scale == 'log':
        up = np.log(up + 0.001)
        down = np.log(np.abs(down) + 0.001)
    elif scale == 'sqrt':
        up = np.sqrt(up)
        down = np.sqrt(np.abs(down))

    _gain = up.abs().rolling(window=period).mean()
    _loss = down.abs().rolling(window=period).mean()

    RS = _gain / _loss
    return 100 - (100 / (1 + RS))

def rsi_cross_ma(df, rsi_days, ma_days, mean_type):
    rsi = RSI(df,rsi_days)
    if mean_type == 'ema':
        rma = rsi.ewm(alpha=1.0 / ma_days, adjust=True).mean()
    elif mean_type == 'sma':
        rma = rsi.rolling(ma_days).mean()
    elif mean_type == 'smm':
        rma = rsi.rolling(ma_days).median()
    else:
        rma = ZLEMA(rsi,ma_days)
    return rsi / rma

def rsi_cross_ma_change(df, rsi_days, ma_days, mean_type, change_days):
    return rsi_cross_ma(df, rsi_days, ma_days, mean_type).pct_change(1).rolling(change_days, closed='left').mean()

def rsi_cross_ma_pctile(df, rsi_days, ma_days, mean_type, pctile_days):
    return moving_percentile(rsi_cross_ma(df, rsi_days, ma_days, mean_type), pctile_days)        

def rsi_slow_rsi_fast_cross(df,fast,slow):
    rsif = RSI(df, fast, meantype='sma')
    rsis = RSI(df, slow, meantype='sma')
    return rsif - rsis

def rsi_slow_rsi_fast_cross_percentile(df,fast,slow,lb):
    rsif = RSI(df, fast, meantype='sma')
    rsis = RSI(df, slow, meantype='sma')
    d = rsif - rsis
    return moving_percentile(d, lb)

def rsi_bollinger(df, rsi_days, boll_days):
    rsidata = RSI(df, rsi_days)
    rsiboll = PERCENT_B(rsidata, boll_days)
    return rsiboll

# ## BBANDS


def BBANDS(ohlc, period, MA=None, column='Close', std_multiplier=2):
    std = ohlc[column].rolling(window=period).std()

    if not isinstance(MA, pd.core.series.Series):
        middle_band = pd.Series(ZLEMA(ohlc, period), name="BB_MIDDLE")
    else:
        middle_band = pd.Series(MA, name="BB_MIDDLE")

    upper_bb = pd.Series(middle_band + (std_multiplier * std), name="BB_UPPER")
    lower_bb = pd.Series(middle_band - (std_multiplier * std), name="BB_LOWER")

    return pd.concat([upper_bb, middle_band, lower_bb], axis=1)

def BBWIDTH(ohlc, period= 20, MA= None, column= "Close"):
    BB = BBANDS(ohlc, period, MA, column)

    return pd.Series(
        (BB["BB_UPPER"] - BB["BB_LOWER"]) / BB["BB_MIDDLE"],
        name="{0} period BBWITH".format(period),
    )

def BBWIDTHPCTILE(ohlc, period= 20, days=20, MA= None, column= "Close"):
    BB = BBWIDTH(ohlc, period, MA, column)

    return moving_percentile(BB, days)

def PERCENT_B(ohlc, period, MA=None, column="Close"):
    BB = BBANDS(ohlc, period, MA, column)
    percent_b = pd.Series(
        (ohlc["close"] - BB["BB_LOWER"]) / (BB["BB_UPPER"] - BB["BB_LOWER"]),
        name="%b",
    )

    return percent_b

# ## Stochastic Oscillator


def STOCH(ohlc, period):
    highest_high = ohlc["High"].rolling(center=False, window=period).max()
    lowest_low = ohlc["Low"].rolling(center=False, window=period).min()

    return pd.Series(
        (ohlc["Close"] - lowest_low) / (highest_high - lowest_low) * 100,
        name="{0} period STOCH %K".format(period),
    )

def STOCHD(ohlc, period, stoch_period):
    return pd.Series(
        STOCH(ohlc, stoch_period).rolling(center=False, window=period).mean(),
        name="{0} period STOCH %D.".format(period),
    )

def STOCHRSI(ohlc, rsi_period, stoch_period):
    rsi = RSI(ohlc, rsi_period)
    return pd.Series(
        ((rsi - rsi.min()) / (rsi.max() - rsi.min()))
        .rolling(window=stoch_period)
        .mean(),
        name="{0} period stochastic RSI.".format(rsi_period),
    )

# ## CCI


def CCI(ohlc, period, constant):
    tp = TP(ohlc)
    tp_rolling = tp.rolling(window=period, min_periods=0)
    mad = tp_rolling.apply(lambda s: abs(s - s.mean()).mean(), raw=True)
    return pd.Series(
        (tp - tp_rolling.mean()) / (constant * mad),
        name="{0} period CCI".format(period),
    )

# ## PowerLevel


def PowerLevel(ohlc, day1, day2):
    bull_power = ohlc["High"] - ZLEMA(ohlc.Close, day1)
    bear_power = ZLEMA(ohlc.Close, day2) - ohlc["Low"]
    return bull_power, bear_power
    
def PowV(v,l):
    return np.sign(v) * np.power(np.abs(v),l)

def DivPower(bull_power, bear_power, l):
    return PowV(bull_power, l) / PowV(bear_power, l)

def PowerEBBP(df, day1, day2, l):
    bull_power, bear_power = PowerLevel(df, day1, day2)
    return DivPower(bull_power, bear_power, l)

def BBP(df, day1, day2, lookback, l):
    bull_power, bear_power = PowerLevel(df, day1, day2)
    dp = DivPower(bull_power, bear_power, l)
    return moving_percentile(dp, lookback)

# ## Gambler's Fallacy


def gamblers_fallacy_intra(df, d1, d2):
    d1t = (df.Close < df.Open).astype(float).rolling(d1).mean()
    d2t = (df.Close < df.Open).astype(float).rolling(d2).mean()
    return d1t / np.power(d2t + 1e-5, 1.25)

def gamblers_fallacy_inter(df, d1, d2):
    d1t = (df.Close < df.Close.shift(1)).astype(float).rolling(d1).mean()
    d2t = (df.Close < df.Close.shift(1)).astype(float).rolling(d2).mean()
    return d1t / np.power(d2t + 1e-5, 1.25)
