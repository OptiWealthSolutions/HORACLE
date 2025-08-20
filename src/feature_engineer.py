import pandas as pd
import numpy as np
import fredapi 
from dotenv import load_dotenv
load_dotenv()

class FeatureEngineer:
    def __init__(self):
        pass

    def tech_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #rsi,sma,return lag

        return df

    def macro_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #interest rate, cpi,ppi,GDP for Japan,USA,Euro Area and Oceania
        fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))
        return df

    def quant_features_engineer (df: pd.DataFrame) -> pd.DataFrame:
        #momentum,volatility,spread ,
        return df
