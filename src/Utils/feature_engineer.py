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


class Tech_FeatureEngineer:
    def __init__(self):
        pass

    def SMA(self, df: pd.DataFrame, period) -> pd.DataFrame:
        # moving average features
        df[f'Mov_av_{period}'] = df['Close'].rolling(window=period).mean()
        return df

    def RSI(self, df: pd.DataFrame, period) -> pd.DataFrame:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df[f'RSI_{period}'] = rsi
        return df

    def LAG_RETURN(self, df, lags):
        for n in lags:
            df[f'RETURN_LAG_{n}'] = df['Close'].pct_change(periods=n)
        return df

    def MACD(self, df):
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd_line
        df['MACD_signal'] = signal_line
        return df

    def STOCH(self, df):
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        stoch = 100 * (df['Close'] - low14) / (high14 - low14)
        df['STOCH'] = stoch
        return df
    

class Macro_FeatureEngineer:
    def interest_rate(self,df: pd.DataFrame) -> pd.DataFrame:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        return df
    def cpi(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def ppi(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def gdp(self,df: pd.DataFrame) -> pd.DataFrame:
        return df
    def sentiment(self,df: pd.DataFrame) -> pd.DataFrame:
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
