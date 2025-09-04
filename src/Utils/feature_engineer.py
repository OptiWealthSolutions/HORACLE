import pandas as pd
import numpy as np
import os 
from dotenv import load_dotenv
from fredapi import Fred
load_dotenv()

# Feature Engineering

# Normalisation : standardiser les features pour éviter les biais d'échelle
# Windowing : utiliser différentes fenêtres temporelles (court, moyen, long terme)
# Lag features : inclure des retards pour capturer la persistence
# Rolling statistics : statistiques glissantes sur différentes périodes

class FeatureEngineer:

    def __init__(self):
        pass
    def features_preparation(df):
        #differeciation 
        df = df.diff().dropna()
        #normalisation
        df = (df - df.mean()) / df.std()
        #windowing (informative)
        return df
    
    def stationarity_test(features_list):
        for el in features_list:
            adfuller_result = adfuller(el)
            p_value = adfuller_result[el][1]
            print(p_value)
        return adfuller_result


        
class Tech_FeatureEngineer:
    def __init__(self):
        pass

    def getSMA(self, df: pd.DataFrame, period) -> pd.DataFrame:
        # moving average features
        df[f'Mov_av_{period}'] = df['Close'].rolling(window=period).mean()
        return df

    def getRSI(self, df: pd.DataFrame, period) -> pd.DataFrame:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df[f'RSI_{period}'] = rsi
        return df



    def getMACD(self, df):
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd_line
        df['MACD_signal'] = signal_line
        return df

    def getSTOCH(self, df):
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        stoch = 100 * (df['Close'] - low14) / (high14 - low14)
        df['STOCH'] = stoch
        return df
    

class Macro_FeatureEngineer:
    def getInterest_rate(self,df: pd.DataFrame) -> pd.DataFrame:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        return df
    def getCPI(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def getPPi(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def getGDP(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def getSentiment(self,df: pd.DataFrame) -> pd.DataFrame:
        return df

class Quant_FeatureEngineer:
    # Volatility clustering : clustering de volatilité
    # VIX-related features : indicateurs liés à la peur du marché
    # ATR (Average True Range) : plage vraie moyenne
    def getVol_clustering (df: pd.DataFrame) -> pd.DataFrame:
        return df

    def getVix_based (df: pd.DataFrame) -> pd.DataFrame:
        return df

    def getATR (df: pd.DataFrame) -> pd.DataFrame:
        return df

    def getLAG_RETURN(self, df, lags):
        for n in lags:
            df[f'RETURN_LAG_{n}'] = df['Close'].pct_change(periods=n)
        return df

    def getPriceMomentum(df):
        return 
    def get12MonthReturn(df):
        return 
    def getPriceAcceleration(df):
        return
    def getMomentumFactors(df,n):
        for lag in range(1, n):
            df[f'momentum_{lag}'] = df['Close'].pct_change(periods=lag)
        return df

    def getMomentumPeriod(df,n):
        prices = df['Close']
        monthly_prices = prices.resample('M').last()
        for lag in range(1, n):
            df[f'return_{lag}m'] = (monthly_prices
            .pct_change(lag)
            .stack()
            .pipe(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
            .add(1)
            .pow(1/lag)
            .sub(1))
        return df


        
    # --- time indicator ---