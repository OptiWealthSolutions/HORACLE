from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

class DataEngineer:
    def __init__(self):
        self.df = None
        pass

    def data_loader (ticker: str, period: str = "15y") -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        return df

    def data_cleaner_close (self):
        #deleting of extrems values
        df = self.df['Close'].copy()
        quantile = df.quantile([0.1, 0.9])
        df = df[(df >= quantile[0.1]) & (df <= quantile[0.9])]
        #fillna
        df = df.fillna(method='ffill')
        return df

    def data_cleaner_returns(self):
        #deleting of extrems values
        df = self.df['Close'].pct_change().dropna()
        quantile = df.quantile([0.01, 0.99])
        df = df[(df >= quantile[0.01]) & (df <= quantile[0.99])]
        #fillna
        df = df.fillna(method='ffill')
        return df

    def standarisation_reg (self):
        df = self.df.copy()
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df

    def standardisation_returns(self):
        df = self.df['Close'].pct_change().dropna()
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df
    
    def normalisation(self,feature_name: str):
        #Normalisation (min-max scaling) : (x - min) / (max - min), utile pour contraindre entre [0,1].
        df = self.df[f'{feature_name}']   
        df = (df - df.min()) / (df.max() - df.min())
        return df

    def stationarity_test(self, feature_name: str, significance=0.05):
        series = self.df[feature_name].dropna()
        result = adfuller(series)
        p_value = result[1]
        stationary = p_value < significance
        return {"p_value": p_value, "stationary": stationary}

class DataProcessing:
    def __init__(self):
        self.df = None
        pass
    
    def heatmap_features(self, features_list):
        df = self.df[f'{features_list}']
        return sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    def colinearity(self, features_list):
        df = self.df[f'{features_list}']
        return df.corr()
    
    def vif_test(self, features_list):
        df = self.df[features_list].dropna()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                           for i in range(len(df.columns))]
        return vif_data

    def ensure_homogeneity(self, features_list):
        df = self.df[features_list]
        summary = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std()
        })
        return summary

    def check_time_structure(self):
        index = self.df.index
        is_datetime = isinstance(index, pd.DatetimeIndex)
        is_sorted = index.is_monotonic_increasing
        no_duplicates = not index.has_duplicates
        return {
            "is_datetime": is_datetime,
            "is_sorted": is_sorted,
            "no_duplicates": no_duplicates
        }