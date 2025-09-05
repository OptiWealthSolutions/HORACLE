import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression

#ML workflow :
#Collect Clean and Validate DATA --> Extract & Engineer Features --> Labelling --> 
# Decide ML model --> Cross Validation & Model Design & Hyper params --> Deploy and predict --> 

#what type of outcome ? ---> Mutliclass (-2,-1,0,1,2)
#what type of features ? ---> Technical, Macro, Quant
#what type of model ? ---> Random Forest Classifier
#labelling method --> rand_forest_labelling_threshold (cf: labelling_engineer)
#cross validation --> Time series cross validation
#hyper params --> Grid search


#What problem we want to solve ?
# We want to predict the future direction and strenght of the price's return for a week & month horizon 

tickerlist = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCAD=X","AUDUSD=X","NZDUSD=X"]

# --- Data Loading, Cleaning and processing ---
def getDataLoad(ticker,period,interval):
    df = yf.download(ticker, period=period, progress=False,interval=interval)
    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")
    df = df['Close'].copy()
    df = df.copy()
    df = df.dropna()
    return df

# --- Features Creating, Engineering, Comparaison and Selection ---
# Linearity, Normality, 
# Homoscedasticity,
# Stationarity, 
def getDataStationarityTest(df, feature_name: str, significance=0.05):
    series = df[feature_name].dropna()
    result = adfuller(series)
    p_value = result[1]
    stationary = p_value < significance
    return {"p_value": p_value, "stationary": stationary}

# Multicollinearity 
#covariance matrix for testing colinearity

# --- Labelling Engineering ---
# Mutliclass Threshold Labelling and Triple barrier labelling
def getDailyVol(df):
    df = df['Close'].index.searchsorted(df['Close'].index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = (pd.Series(df.index[df - 1], 
                   index=df.index[df.shape[0] - df.shape[0]:]))
    
    df = df.loc[df.index] / df.loc[df.values].values - 1  # daily rets
    df = df.ewm(span=span).std()
    return df

def getEvents(df, threshold):
    close = df['Close']
    t_events, s_pos, s_neg = [], 0, 0
    diff = np.log(close).diff().dropna()
    
    for i in diff.index[1:]:
        pos, neg = float(s_pos + diff.loc[i]), float(s_neg + diff.loc[i])
        s_pos, s_neg = max(0.0, pos), min(0.0, neg)
        
        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)
    
    return pd.DatetimeIndex(t_events)

def getSingleBarrier(close, loc, t1, pt, sl):
    t0 = loc  # Event start time
    prices = close[loc:t1]  # Prix sur la période
    
    if len(prices) < 2:
        return pd.NaT, np.nan
    
    cum_rets = (prices / close[t0] - 1.0)
    
    for timestamp, ret in cum_rets.items():
        if timestamp == t0:
            continue
            
        if pd.notna(pt[t0]) and ret >= pt[t0]:
            return timestamp, 1  # Profit taking
        
        if pd.notna(sl[t0]) and ret <= sl[t0]:
            return timestamp, -1  # Stop loss
    
    return t1, 0  # Time barrier



def getTripleBarrierLabels(events, min_ret):
    #min_return_ajd adapted to the volatility
    close = events['Close']
    events['return_std'] = close.pct_change().std()
    events['return_mean'] = close.pct_change().mean()
    min_return_adj = events['return_mean'] * (2*events['return_std'])
    bins = events['ret'].copy()
    bins[bins >= min_return_adj] = 1  # Profit
    bins[bins <= -min_return_adj] = -1  # Loss
    bins[(bins < min_return_adj) & (bins > -min_return_adj)] = 0  # Neutral
    
    return bins
# --- Meta-Labelling ---
def getMetaLabel():
    
    return

# --- Model(s) Building ---
#Primary Model : "Random Forest Classifier"


#Meta-labelling Model


# --- Model(s) Training, Testing, Validation ---
# train-test split

#cross-validation

#hyper params

# --- Model(s) measurement --- 
#classification report
#confusion matrix

# --- BackTesting ---


# --- Main function activation ---

if __name__ == "__main__":
    pass