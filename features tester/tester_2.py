import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Parameters 
PERIOD = "15y"
INTERVAL = "1d"
SMOOTHING_WINDOW = 14
LONG_WINDOW = 51
SHORT_WINDOW = 2
SHIFT = 5

def load_data(ticker):
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    data['Price_z_score'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
    return data

def labelling(df):  
    df[f'Price_z_score+{SHIFT}'] = df['Price_z_score'].shift(-SHIFT)
    df['TARGET'] = df[f'Price_z_score+{SHIFT}'] - df['Price_z_score']
    return df

def add_return_lag(df):
    for i in range(1, SHIFT+1):
        df[f'Lag_{i}'] = df['Close'].diff(i)
    return df
    
def add_sma_features(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    return df 

def add_vol_features(df):
    df['Vol'] = df['Close'].rolling(SHIFT).std()
    return df
def test_regression(df, x, y):
    subset = df[[x, y]].dropna()

    X = subset[[x]].values
    Y = subset[y].values

    model = LinearRegression()
    model.fit(X, Y)
    
    y_pred = model.predict(X)
    r2 = r2_score(Y, y_pred)

    print(f"R2: {r2:.4f}")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Scatter plot avec droite de régression
    plt.figure()
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X, y_pred, color="red")
    plt.title(f"Linear Regression {x} vs {y}")
    plt.show()

    return r2, model.coef_, model.intercept_
def main():
    df = load_data('EURUSD=X')
    df = labelling(df)
    df = add_return_lag(df)
    df = add_sma_features(df)
    df = add_vol_features(df)
    test_regression(df, x='Vol', y='TARGET')

main()
