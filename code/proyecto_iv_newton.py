#%%
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega
from py_vollib.black_scholes.implied_volatility import implied_volatility
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %%
def implied_vol(S0, K, T, r, market_price, flag='c', tol=0.00001):
    """Calculating the implied volatility of an European option
        S0: stock price
        K: strike price
        T: time to maturity
        r: risk-free rate
        market_price: option price in market
    """
    max_iter = 200 #max no. of iterations
    vol_old = 0.3 #initial guess 

    for k in range(max_iter):
        bs_price = bs(flag, S0, K, T, r, vol_old)
        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price

        vol_new = vol_old - C/Cprime
        new_bs_price = bs(flag, S0, K, T, r, vol_new)
        if (abs(vol_old-vol_new) < tol or abs(new_bs_price-market_price) < tol):
            break

        vol_old = vol_new

    implied_vol = vol_new
    return implied_vol
#%%
S0, K, t, r = 83.11, 80, 1/250, 0.025
market_price = 3.23
iv, iter = implied_vol(S0, K, t, r, market_price)
print("La volatilidad implicita {0:5.2f}, fue calculada con {1:.0f}".format(iv*100, iter))

# %%
bs_over_iv = [ bs('c', S0, K, t, r, iv/100) for iv in range(0,100,1) ]
plt.figure()
plt.plot(bs_over_iv)
plt.title('BS prima')
plt.xlabel('Implied Volatility (%)')
plt.ylabel('Call Price ($)')
plt.show()

#%%
def implied_vol2(S0, K, t, r, market_price, flag='c', exa=0.00001, vol_old=0.3, max_iter=200):
    """Calculating the implied volatility of an European option
        S0: stock price
        K: strike price
        T: time to maturity
        r: risk-free rate
        market_price: option price in market
        flag: c or p
        acc: accuracy / error tolerance
        vol_old: initial guess
        max_iter: max no. of iterations
    """
    err_vol = float('inf')
    err_prc = float('inf')
    iter = 0
    bs_price = bs(flag, S0, K, t, r, vol_old)
    while err_vol > exa or err_prc > exa or iter > max_iter:
        Cprime = vega(flag, S0, K, t, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        new_bs_price = bs(flag, S0, K, t, r, vol_new)
        err_vol = abs(vol_old - vol_new)
        err_prc = abs(new_bs_price - market_price)
        vol_old = vol_new
        bs_price = new_bs_price
        iter += 1

    implied_vol = vol_new
    return implied_vol

S0, K, t, r = 83.11, 80, 1/250, 0.025
market_price = 5
#iv, iter = implied_vol(S0, K, t, r, market_price)
#print("La volatilidad implicita {0:5.2f}, fue calculada con {1:.0f}".format(iv*100, iter))

#%%

data = pd.read_csv('C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos'
                     '/idi_ii/tsla_options_last.csv')
data.head()
test = data[10:11]
# %%
S = 1132
r = 0.0025
#%%
test['iv'] = test.apply(lambda row: implied_vol(S,
                                                row['Strike'],
                                                row['tau'],
                                                r,
                                                row['Last Sale']), axis=1)
# %%

test['iv'] = test.apply(lambda row: implied_volatility(row['Last Sale'],
                                                        S,
                                                        row['Strike'],
                                                        row['tau'],
                                                        r,
                                                        row['type']), axis=1)

#%%
iv_vctr = [implied_volatility(row[2],
                                S,
                                row[7],
                                row[8],
                                r,
                                row[9]) for row in test]

#%%
iv_vctr = []
for index, row in test.iterrows():
    print(bs(row['type'], S, row['Strike'], row['tau'], r, 0))
    print(row['Last Sale'])
    if bs(row['type'], S, row['Strike'], row['tau'], r, 0) < row['Last Sale']:
        iv_vctr.append(implied_volatility(row['Last Sale'],S,row['Strike'],row['tau'],r,row['type']))
    else:
        iv_vctr.append(0)
#%%

iv_vctr_be_rational = []
for index, row in data.iterrows():
    if bs(row['type'], S, row['Strike'], row['tau'], r, 0) < row['Last Sale']:
        try:
            iv_vctr_be_rational.append(implied_volatility(row['Last Sale'],S,row['Strike'],row['tau'],r,row['type']))
        except:
            iv_vctr_be_rational.append(0)
    else:
        iv_vctr_be_rational.append(0)

#%%
iv_vctr_newton = []
for index, row in data.iterrows():
    print(index)
    try:
        if bs(row['type'], S, row['Strike'], row['tau'], r, 0) < row['Last Sale']:
            try:
                iv_vctr_newton.append(implied_vol2(S,row['Strike'],row['tau'],r, row['Last Sale'], flag=row['type']))
            except:
                iv_vctr_newton.append(0)
        else:
            iv_vctr_newton.append(0)
    except:
        iv_vctr_newton.append(0)

#%%
data['iv'] = iv_vctr_be_rational
#%%
sns.relplot(
    data=data[data.iv != 0], x='Strike', y='iv', hue='type', col='Expiration Date', kind='scatter', col_wrap=3
).set(ylim=(0, 5))

#%%
implied_volatility(587.5, S, 550, 0.0083, r, 'c')
implied_vol(S, 550, 0.0083, r, 587,)
# %%
bs('c', S,550, 0.0083, r,4)
# %%
