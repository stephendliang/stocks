
# ## Stochastic Oscillator


def test_stoch(df):
    mac = np.zeros(shape=(200))
    data2 = incf3[-60:-8]

    for d1 in tqdm(range(1,201)):
        data1 = STOCH(df,d1)
        dx = data1[-60:-8]
        corr, b = pearsonr(dx, data2)
        mac[(d1-1)] = corr*corr

    return mac

def test_stochd(df):
    mac = np.zeros(shape=(100,250))
    data2 = incf3[-60:-8]

    for d1 in tqdm(range(2,102)):
        for d2 in range(1,251):
            data1 = STOCHD(df,d1,d2)
            dx = data1[-60:-8]
            corr, b = pearsonr(dx, data2)
            mac[(d1-2),(d2-1)] = corr*corr

    return mac

def test_stochrsi(df):
    mac = np.zeros(shape=(100,250))
    data2 = incf3[-60:-8]

    for d1 in tqdm(range(1,101)):
        for d2 in range(1,251):
            data1 = STOCHRSI(df,d1,d2)
            dx = data1[-60:-8]
            if (np.isin(True, np.isinf(dx)) == False and np.isin(True, np.isnan(dx))  == False):
                corr, b = pearsonr(dx, data2)
                mac[(d1-1),(d2-1)] = corr*corr

    return mac

stochres = test_stoch(semiconductor_dfs['SOXX'])
print('stoch', np.nanmax(stochres), np.unravel_index(np.nanargmax(stochres), stochres.shape))

stochdres = test_stochd(semiconductor_dfs['SOXX'])
print('stochd', np.nanmax(stochdres), np.unravel_index(np.nanargmax(stochdres), stochdres.shape))

stochrsires = test_stochrsi(semiconductor_dfs['SOXX'])
print('stochrsi', np.nanmax(stochrsires), np.unravel_index(np.nanargmax(stochrsires), stochrsires.shape))

# ## RSI / MA(RSI) section


meantypes = ['sma','smm','ema','zlema']
scaletype = ['linear','sqrt']

def test_rsi_cross_ma(df):
    mac = np.zeros(shape=(len(meantypes),320,200))
    data2 = incf3[-60:-8]

    for idxm, meantype in enumerate(meantypes):
        for rsi_days in tqdm(range(2,312)):
            for ma_days in range(3,156):
                data1 = rsi_cross_ma(df, rsi_days, ma_days, meantype)
                dx = data1[-60:-8]
                if np.isin(True, np.isinf(dx)) == False and np.isin(True, np.isnan(dx)) == False:
                    corr, b = pearsonr(dx, data2)
                    mac[idxm,(rsi_days-2),(ma_days-3)] = corr*corr

    return mac

ma_cross_soxx = test_rsi_cross_ma(semiconductor_dfs['SOXX'])

print('RSI MA Cross', np.nanmax(ma_crossv_soxx), np.unravel_index(np.nanargmax(ma_crossv_soxx), ma_crossv_soxx.shape))
plt.imshow(ma_cross_soxx[0,1])

# ## RSI/MA Velocity


def test_rsi_sub_ma_vel(df):
    mac = np.zeros(shape=(len(scaletype),len(meantypes),310,112,135))
    data2 = incf3[-60:-8]

    for idxs, sc in enumerate(scaletype[:1]):
        for idxm, mt in enumerate(meantypes[:1]):
            for d1 in tqdm(range(2,312,2)):
                for d2 in range(3,115,2):
                    for d3 in range(1,30):
                        data1 = rsi_cross_ma_change(df,d1,d2,d3,'div',mt,sc)
                        dx = data1[-60:-8]
                        if np.isin(True, np.isinf(dx)) == False and np.isin(True, np.isnan(dx)) == False:
                            corr, b = pearsonr(dx, data2)
                            mac[idxs,idxm,(d1-2),(d2-3),(d3-1)] = corr*corr

    return mac

ma_crossv_soxx  = test_rsi_sub_ma_vel(semiconductor_dfs['SOXX'])
print('RSI MA Cross Velocity', np.nanmax(ma_crossv_soxx[0]), np.unravel_index(np.nanargmax(ma_crossv_soxx[0]), ma_crossv_soxx[0].shape))

plt.imshow(ma_crossv_soxx[0,0,34])

# ### RSI cross (MA(RSI)) Percentile


def test_rsi_sub_ma_pct(df):
    mac = np.zeros(shape=(len(scaletype),len(meantypes),310,12,135))
    data2 = incf3[-60:-8]

    for idxs, sc in enumerate(scaletype):
        for idxm, mt in enumerate(meantypes):
            for d1 in range(2,300,5):
                for d2 in range(3,15,3):
                    for d3 in range(1,30):
                        data1 = rsi_cross_ma_pct(df[-400:],d1,d2,d3,'div',mt,sc)
                        dx = data1[-60:-8]
                        if (np.isin(True, np.isinf(dx)) == False and np.isin(True, np.isnan(dx))  == False):
                            corr, b = pearsonr(dx, data2)
                            mac[idxs,idxm,(d1-2),(d2-3),(d3-1)] = corr*corr

    return mac

mac = test_rsi_sub_ma_pct(semiconductor_dfs['SOXX'])

print('all', np.nanmax(mac), np.unravel_index(np.nanargmax(mac), mac.shape))

# ### RSI cross MA Velocity Percentile


def test_rsi_ma_cross_velocity(lookback):
    cdays = list(range(2,258,2))
    rdays = list(range(256,512,2))
    mdays = list(range(3,21))

    rxm = np.zeros(shape=(128,128,18))

    for lb in tqdm(lookback):
        for dc in tqdm(cdays):
            for d1 in (rdays):
                for d2 in (mdays):
                    data1 = percentile(rsi_cross_ma_change(df,d1,d2,dc), lb)
                    data2 = incf7
                    corr, b = pearsonr(data1[-60:-8], data2[-60:-8])
                    rxm[(dc-2)//2,(d1-256)//2,(d2-6)] = corr*corr

    print(np.max(rxm), np.unravel_index(rxm.argmax(), rxm.shape))

def test_rsi_cross_rsi(df):
    mac = np.zeros(shape=(255,255))
    data2 = incf3[-60:-8]

    for d1 in tqdm(range(2,255)):
        for d2 in range(3,255):
            if d1 != d2:
                data1 = rsi_slow_rsi_fast_cross(df,d1,d2)
                dx = data1[-60:-8]
                corr, b = pearsonr(dx, data2)
                mac[d1,d2] = corr*corr

    return mac

mac = test_rsi_cross_rsi(semiconductor_dfs['SOXX'])

mac[np.unravel_index(np.nanargmax(mac), mac.shape)]

# ## Gamblers Fallacy


gmb = np.zeros(shape=(600,600))
for d1 in tqdm(range(2,602)):
    for d2 in (range(d1 + 1,602)):
        lt15 = (df.Close < df.Open).astype(float).rolling(d1).mean()
        lt30 = (df.Close < df.Open).astype(float).rolling(d2).mean()

        data1 = (lt15) / np.power(lt30 + 1e-5, 1.25) # np.power( + 1e-2, 3)
        data2 = incf7
        
        corr, b = pearsonr(data1[-60:-8], data2[-60:-8])
        gmb[d1-2,d2-2] = corr

print(np.max(gmb), np.unravel_index(gmb.argmax(), gmb.shape))
print(np.min(gmb), np.unravel_index(gmb.argmin(), gmb.shape))

plt.imshow(gmb)

# ## Change Percentile


mcp = np.zeros(shape=(55,100))

for lb in tqdm(range(2,57)):
    for day in (range(1,101)):
        mxi = df.Close.pct_change().rolling(day).mean()
        
        data1 = moving_percentile(mxi,lb)
        data2 = incf7

        corr, b = pearsonr(data1[-60:-8], data2[-60:-8])

        mcp[lb-2,day-1] = corr*corr


print(np.max(mcp), np.unravel_index(mcp.argmax(), mcp.shape))

# ## Reversal


gmb = np.zeros(shape=(100))

for day in (range(1,101)):
    increase  = pd.Series(max_inc_forward(df.High.values,df.Low.values,3))
    decrease  = pd.Series(max_dec_forward(df.High.values, df.Low.values, day)).shift(day)
    #print(decrease[day:])
    #print(increase[:-day])#

    inc = np.maximum(day, 3)
    corr, b = pearsonr(increase[:-inc], decrease[inc:])
    gmb[day-1] = corr * corr

print(np.max(gmb), gmb.argmax())

gmb

plt.plot(gmb)

# # Preparing Data for Training


def 
