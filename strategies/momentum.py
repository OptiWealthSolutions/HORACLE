#package importation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yfinance as yf
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ta
from statsmodels.tsa.stattools import adfuller

class MomentumStrategy():
    def __init__(self):
        self.ticker = "EURUSD=X"
        self.PERIOD = "15y"
        self.INTERVAL = "5d"
        self.SHIFT = 5
        self.lags = [1,2,3,6,9,12]
        self.df = self.getDataLoad()
        self.df = self.getDataFrameClean()
        # self.std_daily = getDailyVol(self.df)  # Removed as getDailyVol is undefined

    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL)
        df = df.dropna()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    def getDataFrameClean(self):
        #deleting of the outlier
        df = self.df.dropna()
        Q1 = df['Close'].quantile(0.10)
        Q3 = df['Close'].quantile(0.90)
        IQR = Q3 - Q1

        # Définir les bornes
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrer les outliers
        df_clean = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
        return df_clean

    # --- features engineering --- 
    def getRSI(self):
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close'], window=14).rsi().to_numpy().reshape(-1)
        return self.df
    
    def PriceMomentum(self):
        self.df['PriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12).roc().to_numpy().reshape(-1)
        return self.df
    
    def getLagReturns(self):
        """Compute lagged log returns for each lag in self.lags.
        This uses log(Close_t / Close_{t-n}) which is the standard lagged log-return.
        """
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = np.log(self.df['Close'] / self.df['Close'].shift(n))
        return self.df
    
    def PriceAccel(self):
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['velocity'] = self.df['log_return']
        self.df['acceleration'] = self.df['log_return'].diff()    
        return self.df
    
    def getPct52WeekHigh(self):
        self.df['52w_high'] = self.df['High'].rolling(window=252).max()
        self.df['Pct52WeekHigh'] = self.df['Close'] / self.df['52w_high']
        return self.df
    
    def getPct52WeekLow(self):
        self.df['52w_low'] = self.df['Low'].rolling(window=252).min()
        self.df['Pct52WeekLow'] = self.df['Close'] / self.df['52w_low']
        return self.df
    
    def get12MonthPriceMomentum(self):
        self.df['12MonthPriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12).roc().to_numpy().reshape(-1)
        return self.df
    
    def getVol(self):
        self.df['MonthlyVol'] = self.df['Close'].pct_change().rolling(window=20).std()
        return self.df

    def getFeaturesDataSet(self):
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close'], axis=1, errors='ignore')
        return self.df_features
    
    #--- statisticals test ---
    def testStationarity(self):
        for col in self.df_features.columns:
            adfuller_result = adfuller(self.df_features[col].dropna())
            print(f"Stationarity test for {col}: {adfuller_result}")
        return
    
    def getCorr(self):
        return
    
    def getFeatureImportance(self):
        return
    
    def getFeatureSelection(self):
        return
    #---  labels engineering ---

    def getLabels(self):
        thresold = self.df['log_return'].std() * 1.2
        self.df['Label'] = np.where(self.df['log_return'] > thresold, 1, 
            np.where(self.df['log_return'] < -thresold, -1, 0))
        return self.df

    #--- model training ---
    def RandomForest(self):
        self.df = self.df.dropna(subset=['Label'])
        self.df_features = self.df_features.loc[self.df.index.intersection(self.df_features.index)]
        X = self.df_features.drop('Label', axis=1, errors='ignore').values   
        y = self.df['Label'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        PrimaryModel =  RandomForestClassifier(
            n_estimators=300,      # 300 arbres
            max_depth=10,          # Profondeur max = 10
            #min_samples_split=50,  # Min 50 échantillons pour split
            #min_samples_leaf=20,   # Min 20 échantillons par feuille
            #max_features='sqrt',   # √(nb_features) features par split
            #bootstrap=True,        # Bootstrap sampling
            #oob_score=True,        # Out-of-bag score
            class_weight='balanced', # Équilibrer les classes
            random_state=42
        )
        PrimaryModel.fit(X_train, y_train)
        y_pred = PrimaryModel.predict(X_test)

        #metrics
        confusion_matrix_ = confusion_matrix(y_test,PrimaryModel.predict(X_test))
        print(f"Confusion matrix: {confusion_matrix_}")
        classification_report_ = classification_report(y_test,PrimaryModel.predict(X_test))
        print(f"Classification report: {classification_report_}")

        return

def main():
    ms = MomentumStrategy()
    # __init__ already loads and cleans data; now compute features and labels
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekHigh()
    ms.getPct52WeekLow()
    ms.get12MonthPriceMomentum()
    ms.getVol()
    ms.getFeaturesDataSet()
    ms.testStationarity()
    ms.getCorr()
    ms.getFeatureImportance()
    ms.getFeatureSelection()
    ms.getLabels()
    ms.RandomForest()
    return

if __name__ == "__main__":
    main()