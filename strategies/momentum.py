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

class MomentumStrategy():
    def __init__(self):
        self.ticker = "EURUSD=X"
        self.df = getDataLoad(self.ticker)
        self.df = getDataFrameClean(self.df)
        self.PERIOD = "15y"
        self.INTERVAL = "1w"
        self.SHIFT = 5
        self.lags = [1,2,3,6,9,12]
        self.std_daily = getDailyVol(self.df)

    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL)
        df = df.copy()
        df = df.dropna()
        df['log_return'] = np.log(df['Close'].pct_change())
        return df

    def getDataFrameClean(self):
        #deleting of the outlier
        df = self.df.dropna()
        Q1 = df['feature'].quantile(0.10)
        Q3 = df['feature'].quantile(0.90)
        IQR = Q3 - Q1

        # Définir les bornes
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrer les outliers
        df_clean = df[(df['feature'] >= lower_bound) & (df['feature'] <= upper_bound)]
        return df_clean

    # --- features engineering --- 
    def getRSI(self):
        self.df['RSI'] = self.df['Close'].pct_change().rolling(window=14).apply(lambda x: 100 - (100 / (1 + np.exp(-x))))
        return
    
    def PriceMomentum(self):
        self.df['PriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12, fillna=False)
        return
    
    def getLagReturns(self):
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = np.log(self.df['Close'].pct_change(periods=n))
        return
    
    def PriceAccel(self):
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['velocity'] = self.df['log_return']
        self.df['acceleration'] = self.df['log_return'].diff()    
        return self.df
    
    def getPct52WeekHigh(self):
        self.df['52w_high'] = self.df['High'].rolling(window=252).max()
        last_52w_high = self.df['52w_high'].iloc[-1]
        self.df['Pct52WeekHigh'] = self.df['Close']/last_52w_high
        return self.df
    
    def getPct52WeekLow(self):
        self.df['52w_low'] = self.df['Low'].rolling(window=252).min()
        last_52w_low = self.df['52w_low'].iloc[-1]
        self.df['Pct52WeekLow'] = self.df['Close']/last_52w_low
        return self.df
    
    def get12MonthPriceMomentum(self):
        self.df['12MonthPriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12, fillna=False)
        return self.df
    
    def getVol(self):
        self.df['MonthlyVol'] = self.df['Close'].pct_change().rolling(window=20).std()
        return self.df

    def getFeaturesDataSet(self):
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close'], axis=1)
        return self.df_features
    
    #--- statisticals test ---
    def testStationarity(self):
        for col in self.df_features.columns:
            adfuller_result = adfuller(self.df_features[col])
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
        thresold = self.std_daily * 1.2
        self.df['Label'] = np.where(self.df['Return'].pct_change() > thresold, 1, 
            np.where(self.df['Return'].pct_change() < -thresold, -1, 0))
        return self.df

    #--- model training ---
    def RandomForest(self):
        X = self.df_features.drop('Label', axis=1).values   
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
    MomentumStrategy = MomentumStrategy()
    df = MomentumStrategy.getDataLoad()
    df = MomentumStrategy.getDataFrameClean(df)
    df = MomentumStrategy.getRSI()
    df = MomentumStrategy.PriceMomentum()
    df = MomentumStrategy.getLagReturns()
    df = MomentumStrategy.PriceAccel()
    df = MomentumStrategy.getPct52WeekHigh()
    df = MomentumStrategy.getPct52WeekLow()
    df = MomentumStrategy.get12MonthPriceMomentum()
    df = MomentumStrategy.getVol()
    df = MomentumStrategy.getFeaturesDataSet()
    df = MomentumStrategy.testStationarity()
    df = MomentumStrategy.getCorr()
    df = MomentumStrategy.getFeatureImportance()
    df = MomentumStrategy.getFeatureSelection()
    df = MomentumStrategy.getLabels()
    df = MomentumStrategy.RandomForest()
    return 

main()