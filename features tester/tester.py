import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import brute

#params 
PERIOD = "15y"
INTERVAL = "1mo"

SMOOTHING_WINDOW = 14
LONG_WINDOW = 51
SHORT_WINDOW = 2

SHIFT = 5

#load data 
def load_data(ticker):
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    return data

#create target feature
def add_label(df):
    df['TARGET'] = df['Close'].diff(SHIFT)
    return df

#create sma features
def add_SMA_crossing(df):
    df[f'SMA{LONG_WINDOW}'] = df['Close'].rolling(window=LONG_WINDOW).mean()
    df[f'SMA{SHORT_WINDOW}'] = df['Close'].rolling(window=SHORT_WINDOW).mean()
    df['SIGNAL'] = np.where(
        df[f'SMA{LONG_WINDOW}'] > df[f'SMA{SHORT_WINDOW}'], 1,
        np.where(df[f'SMA{LONG_WINDOW}'] < df[f'SMA{SHORT_WINDOW}'], -1, 0)
    )
    return df

#create lag return features
def add_return_lag(df):
    for lag in range(1,SHIFT+1):
        df[f'RETURN_LAG_{lag}'] = df['Close'].diff(lag)
    return df

#create volatility feature
def add_volatility(df):
    df['VOLATILITY'] = df['Close'].rolling(window=SHIFT).std()
    return df


def objective(params, df, target_col):
    short_w, long_w = int(params[0]), int(params[1])
    if short_w >= long_w:
        return 1  # pénalisation pour fenêtres incohérentes
    
    df[f"SMA_{short_w}"] = df['Close'].rolling(window=short_w).mean()
    df[f"SMA_{long_w}"] = df['Close'].rolling(window=long_w).mean()
    df['SIGNAL'] = np.where(
        df[f'SMA{LONG_WINDOW}'] > df[f'SMA{SHORT_WINDOW}'], 1,
        np.where(df[f'SMA{LONG_WINDOW}'] < df[f'SMA{SHORT_WINDOW}'], -1, 0)
    )
    
    subset = df[[target_col, 'SIGNAL']].dropna()
    if len(subset) == 0:
        return 1
    
    corr = subset[target_col].corr(subset['SIGNAL'])
    return -abs(corr)   # brute cherche à minimiser, on inverse le signe
    
#optimize the couple of sma
def optimize_sma(df, target_col, short_range, long_range):
    rranges = (slice(short_range.start, short_range.stop, 1),
               slice(long_range.start, long_range.stop, 1))
    res = brute(objective, rranges, args=(df, target_col), finish=None)
    best_short, best_long = int(res[0]), int(res[1])
    print(f"Best pair: ({best_short},{best_long})")
    return best_short, best_long

#testing the corr between feature and target with linear regression
def linear_regression(df,x,y):
    subset  = df[[x,y]].dropna()

    X = subset[[x]].values
    y = subset[y].values

    model = LinearRegression()
    model.fit(X,y)

    y_pred = model.predict(X)
    r2 = r2_score(y,y_pred)
    coefficients = model.coef_
    intercept = model.intercept_
    print(f"R2: {r2}")
    
    print(f"Coefficients: {coefficients}")
    print(f"Intercept: {intercept}")

    plt.figure()
    plt.title(f"Linear Regression {x} vs Target")
    plt.scatter(X,y)
    plt.show()
    return r2, coefficients, intercept 

#main function for execute all the code
def main():
    df = load_data("AAPL")
    df = add_label(df)
    list_of_features = ['RETURN_LAG_1','RETURN_LAG_2','RETURN_LAG_3','RETURN_LAG_4','RETURN_LAG_5','SMA2','SMA51','VOLATILITY']
    df = add_SMA_crossing(df)
    df = add_return_lag(df)
    df = add_volatility(df)
    optimize_sma(df,'TARGET',range(2,50),range(51,201))
    for el in list_of_features:
        linear_regression(df,el,'TARGET')
    return  

df = main()