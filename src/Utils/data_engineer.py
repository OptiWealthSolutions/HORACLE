import yfinance as yf
import pandas as pd

class DataEngineer:
    def __init__(self):

        pass

    def data_loader (ticker: str, period: str = "15y") -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        return df

    def data_cleaner (df: pd.DataFrame) -> pd.DataFrame:
        #deleting of extrems values
        df = df['Close'].copy()
        quantile = df.quantile([0.1, 0.9])
        df = df[(df >= quantile[0.1]) | (df <= quantile[0.9])]
        #fillna
        df = df.fillna(method='ffill')
        return df
    def standarisation (df: pd.DataFrame) -> pd.DataFrame:
        return df
    