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
        df['return'] = df['Close'].shift(-self.SHIFT)
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
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close'], window=14).rsi()
        return self.df
    
    def PriceMomentum(self):
        self.df['PriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12).roc().to_numpy()
        return self.df
    
    def getLagReturns(self):
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = np.log(self.df['Close'] / self.df['Close'].shift(n))
        return self.df
    
    def PriceAccel(self):
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['velocity'] = self.df['log_return']
        self.df['acceleration'] = self.df['log_return'].diff()    
        return self.df
    
    def getPct52WeekHigh(self):
        w_high = self.df['High'].rolling(window=252).max()
        self.df['Pct52WeekHigh'] = self.df['Close'] / w_high
        return self.df
    
    def getPct52WeekLow(self):
        w_low = self.df['Low'].rolling(window=252).min()
        self.df['Pct52WeekLow'] = self.df['Close'] / w_low
        return self.df
    
    def get12MonthPriceMomentum(self):
        self.df['12MonthPriceMomentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=12).roc()
        return self.df
    
    def getVol(self):
        self.df['MonthlyVol'] = self.df['Close'].pct_change().rolling(window=20).std()
        return self.df

    def getFeaturesDataSet(self):
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close'], axis=1, errors='ignore')
        self.df_features
        print(self.df_features)
        return self.df_features
    
    #--- statisticals test ---
    def testStationarity(self):
        for col in self.df_features.columns:
            adfuller_result = adfuller(self.df_features[col].dropna())
            print(f"Stationarity test for {col}: {adfuller_result}")
        return
    
    # def getCorr(self):
    #     return
    
    # def getFeatureImportance(self):
    #     return
    
    # def getFeatureSelection(self):
    #     return

    #---  labels engineering ---
   

    def getLabels(self, seuil_lambda=0.1):
        # DIAGNOSTIC 1: Vérifier les données de base
        print(f"Statistiques de la colonne 'return':")
        print(self.df['return'].describe())
        print(f"\nNombre de NaN: {self.df['return'].isna().sum()}")
        
        # DIAGNOSTIC 2: Vérifier le seuil par rapport aux données
        print(f"\nSeuil utilisé: ±{seuil_lambda}")
        print(f"Valeurs > {seuil_lambda}: {(self.df['return'] > seuil_lambda).sum()}")
        print(f"Valeurs < -{seuil_lambda}: {(self.df['return'] < -seuil_lambda).sum()}")
        print(f"Valeurs entre ±{seuil_lambda}: {((self.df['return'] >= -seuil_lambda) & (self.df['return'] <= seuil_lambda)).sum()}")
        
        # DIAGNOSTIC 3: Proposer des seuils alternatifs basés sur les quantiles
        q95 = self.df['return'].quantile(0.95)
        q05 = self.df['return'].quantile(0.05)
        std_return = self.df['return'].std()
        
        print(f"\nSeuils alternatifs suggérés:")
        print(f"  95e percentile: {q95:.4f}")
        print(f"  5e percentile: {q05:.4f}")
        print(f"  1 écart-type: ±{std_return:.4f}")
        print(f"  0.5 écart-type: ±{std_return*0.5:.4f}")
        
        # Créer les labels avec le seuil donné
        self.df['TARGET'] = np.where(self.df['return'] > seuil_lambda, 1, 
                                    np.where(self.df['return'] < -seuil_lambda, -1, 0))
        
        # DIAGNOSTIC 4: Distribution finale
        print(f"\nDistribution des labels avec seuil {seuil_lambda}:")
        print(self.df['TARGET'].value_counts().sort_index())
        
        # Sauvegarder
        self.df['TARGET'].to_csv('labels.csv', index=False)
        
        return self.df
        
    #--- model training ---
    def RandomForest(self):
        X = self.df_features.values 
        y = self.df['TARGET'].values
        print(self.df)
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
    #ms.getRSI()
    #ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekHigh()
    ms.getPct52WeekLow()
    #ms.get12MonthPriceMomentum()
    ms.getVol()
    ms.getFeaturesDataSet()
    #ms.testStationarity()
    #ms.getCorr()
    #ms.getFeatureImportance()
    #ms.getFeatureSelection()
    ms.getLabels()
    ms.RandomForest()
    return

if __name__ == "__main__":
    main()