from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

class DataEngineer :
    def __init__(self):
        pass

    def getDataLoadAll (ticker,period,interval) -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        return df

    def getDataLoad(ticker,period,interval):
        df = yf.download(ticker, period=period, progress=False,interval=interval)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        df = df['Close'].copy()
        return df

    def getDataClean(df):
        df = df.copy()
        df = df.dropna()
        return df

    def getDataStandardized (df):
        df = df.copy()
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df

    def getDataNormalised(df,feature_name: str):
        #Normalisation (min-max scaling) : (x - min) / (max - min), utile pour contraindre entre [0,1].
        df = df[f'{feature_name}']   
        df = (df - df.min()) / (df.max() - df.min())
        return df

    def getDataStationarityTest(df, feature_name: str, significance=0.05):
        series = df[feature_name].dropna()
        result = adfuller(series)
        p_value = result[1]
        stationary = p_value < significance
        return {"p_value": p_value, "stationary": stationary}

    def getDataHeatmap(df, features_list):
        df = df[f'{features_list}']
        return sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    def getDataColinearity(df, features_list):
        df = df[f'{features_list}']
        return df.corr()
        
    def getDataVifTest(df, features_list):
        df = df[features_list].dropna()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                        for i in range(len(df.columns))]
        return vif_data

    def getDataHomogeneity(df, features_list):
        df = df[features_list]
        summary = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std()
        })
        return summary

    def getDataTimeStructure(df):
        index = df.index
        is_datetime = isinstance(index, pd.DatetimeIndex)
        is_sorted = index.is_monotonic_increasing
        no_duplicates = not index.has_duplicates
        return {
                "is_datetime": is_datetime,
                "is_sorted": is_sorted,
                "no_duplicates": no_duplicates
            }

    def getDataMPT_test(df_features):
        #marcenko_pastur theorem
        #q= T/N
        q = len(df_features)/len(df_features.columns)
        #création de la matrice de cov des features
        cov_matrix = df_features.cov()
        #calcule des valeurs propres
        eigenvalues = np.linalg.eig(cov_matrix)[0]
        #calcul des intervalles :
        #this function aims to verifie if more than 1 feature give us the same informations, in this case we can denoising the dataframe
        eMin, eMax = np.var(cov_matrix)*(1- (1./q))**2, np.var(cov_matrix)*(1+ (1./q))**2
        return eigenvalues, eMin, eMax
    
    def getDataPartialDifferentiation(df,features_name):
        df = df.copy()
        df[f'{features_name}_diff'] = np.log(df[f'{features_name}'].diff().dropna())
        return df

    def getDataDifferenciation(df,feature_name):
        df = df.copy()
        df[f'{feature_name}_diff'] = df[f'{feature_name}'].diff().dropna()
        return df