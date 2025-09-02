import pandas as pd
import numpy as np
import fredapi 
from dotenv import load_dotenv
load_dotenv()

# Feature Engineering

# Normalisation : standardiser les features pour éviter les biais d'échelle
# Windowing : utiliser différentes fenêtres temporelles (court, moyen, long terme)
# Lag features : inclure des retards pour capturer la persistence
# Rolling statistics : statistiques glissantes sur différentes périodes

import pandas_ta as ta
class Tech_FeatureEngineer:
    def __init__(self):
        pass

    def sma(df: pd.DataFrame,period) -> pd.DataFrame:
        #rsi,sma,return lag
        #moving average features
        df[f'Mov_av_{period}'] = df['Close'].rolling(period).mean()

    def rsi(df: pd.DataFrame,period) -> pd.DataFrame:
        df[f'RSI_{period}'] = ta.RSI(close=df['Close'], length=period)

    def lag_return(df,lags = List[int]):
        #return lag n for n target horizons we want n-1 lag for targeting data we have not seen before
        for n in lags:
            df[f'RETURN_LAG_{n}'] = df['Close'].diff(n)
        return df
    def macd(df):
        df['MACD'] = ta.MACD(close=df['Close'])
        return df
    
    def rsi(df):
        df['RSI'] = ta.RSI(close=df['Close'])
        return df
    
    def stoch(df):
        df['STOCH'] = ta.STOCH(close=df['Close'])
        return df
    
    

class Macro_FeatureEngineer:
    def interest_rate(df: pd.DataFrame) -> pd.DataFrame:
        fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))
        return df
    def cpi(df: pd.DataFrame) -> pd.DataFrame:
        return df
    def ppi(df: pd.DataFrame) -> pd.DataFrame:
        return df
    def gdp(df: pd.DataFrame) -> pd.DataFrame:
        return df
    def sentiment(df):
        return df
class Quant_FeatureEngineer:
    # Volatility clustering : clustering de volatilité
    # VIX-related features : indicateurs liés à la peur du marché
    # ATR (Average True Range) : plage vraie moyenne
    def vol_clustering (df: pd.DataFrame) -> pd.DataFrame:
        return df
    def vix_based (df: pd.DataFrame) -> pd.DataFrame:
        return df
    def atr (df: pd.DataFrame) -> pd.DataFrame:
        return df

