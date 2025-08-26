import pandas as pd
import numpy as np
import fredapi 
from dotenv import load_dotenv
load_dotenv()
import pandas_ta as ta
class FeatureEngineer:
    def __init__(self):
        pass

    def tech_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #rsi,sma,return lag
        #moving average features
        df['Mov_av_20'] = df['Close'].rolling(20).mean()
        df['Mov_av_50'] = df['Close'].rolling(50).mean()
        df['Mov_av_200'] = df['Close'].rolling(200).mean()

        #RSI feature
        df['RSI'] = ta.RSI(close=df['Close'], length=14)

        #return lag n for n target horizons we want n-1 lag for targeting data we have not seen before
        for n in range(1,6):
            df[f'RETURN_LAG_{n}'] = df['Close'].diff(n)

        return df

    def macro_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #interest rate, cpi,ppi,GDP for Japan,USA,Euro Area and Oceania
        fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))
        return df

    def quant_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #momentum,volatility,spread ,
        return df


