import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Parameters 
PERIOD = "15y"
INTERVAL = "1mo"
SHIFT = 5

def load_data(ticker):
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    return data

def create_target(df, shift=SHIFT):
    df['Shifted'] = df['Close'].shift(-shift)
    df['Target'] = df['Shifted'] - df['Close']
    return df

def add_return_lag(df):
    for i in range(1, SHIFT+1):
        df[f'Lag_{i}'] = df['Close'].diff(i)
    return df

def add_sma_features(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['Crossing_Result'] = np.where(df['SMA20'] > df['SMA50'], 1, 0)
    return df 

def add_vol_features(df):
    df['Vol'] = df['Close'].rolling(SHIFT).std()
    return df

def test_correlation(df):
    for col in ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Vol', 'Crossing_Result']:
        corr = df[[col, 'Target']].corr().iloc[0,1]
        print(f"{col}: correlation avec TARGET = {corr:.3f}")
    return corr

def main():
    df = load_data('EURUSD=X')
    df = create_target(df)
    df = add_return_lag(df)
    df = add_sma_features(df)
    df = add_vol_features(df)
    test_correlation(df)

main()
