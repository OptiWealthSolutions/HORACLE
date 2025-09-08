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
from sklearn.metrics import roc_auc_score
class MomentumStrategy():
    def __init__(self):
        self.ticker = "EURUSD=X"
        self.PERIOD = "20y"
        self.INTERVAL = "1d"
        self.SHIFT = 5
        self.lags = [1,2,3,6,9,12]
        self.df = self.getDataLoad()

        # self.std_daily = getDailyVol(self.df)  # Removed as getDailyVol is undefined

    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL)
        df = df.dropna()
        
        # Log return quotidien
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
        # Rendement futur sur SHIFT jours
        df['return'] = (df['Close'].shift(-self.SHIFT) - df['Close']) / df['Close']
    
        df = df.dropna()  
        Q1 = df['Close'].quantile(0.15)
        Q3 = df['Close'].quantile(0.85)
        IQR = Q3 - Q1

        # Définir les bornes
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrer les outliers
        df_clean = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]# supprimer les lignes NaN introduites par shift
        return df

    # --- features engineering --- 
    def getRSI(self):
        self.df['RSI'] = self.df['Close'].diff().pipe(lambda x: x.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean() / self.df['Close'].diff().pipe(lambda x: -x.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        self.df['RSI'] = 100 - (100 / (1 + self.df['RSI']))
        return self.df
    
    def PriceMomentum(self):
        self.df['PriceMomentum'] = (self.df['Close'] / self.df['Close'].shift(12) - 1) * 100
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
        self.df['12MonthPriceMomentum'] = (self.df['Close'] / self.df['Close'].shift(252) - 1) * 100
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
    
    def getCorr(self):
        return
    
    def getFeatureImportance(self):
        return
    
    def getFeatureSelection(self):
        return

    #---  labels engineering ---
    def getLabels(self, profit_target=0.01, stop_loss=0.01, max_hold_days=5, volatility_scaling=True):
        prices = self.df['Close']
        n = len(prices)
        
        # Calcul de la volatilité roulante si scaling activé
        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std().fillna(returns.std())
        
        prices_array = prices.values
        labels = np.zeros(n)
        entry_dates, exit_dates, entry_prices, exit_prices, returns_pct, hold_days, barrier_hit, vol_adj_arr = [], [], [], [], [], [], [], []


        def _find_first_barrier_hit(prices, entry_idx, profit_target, stop_loss, max_hold):
            entry_price = prices[entry_idx]
            end_idx = min(entry_idx + max_hold, len(prices) - 1)
            for i in range(entry_idx + 1, end_idx + 1):
                ret = (prices[i] - entry_price) / entry_price
                if ret >= profit_target:
                    return 1, i
                elif ret <= -stop_loss:
                    return -1, i
            return 0, end_idx

        for i in range(n):
            # Ajustement des barrières selon volatilité
            if volatility_scaling:
    # Remplissage des NaN
                vol_filled = vol.fillna(method='bfill').fillna(method='ffill')
                # Forcer la valeur scalaire
                vol_value = float(vol_filled.iloc[i])
                vol_adj = max(vol_value / 0.02, 0.5)
                profit_adj = profit_target * vol_adj
                loss_adj = stop_loss * vol_adj
            else:
                profit_adj = profit_target
                loss_adj = stop_loss
                vol_adj = 1.0

            label, exit_idx = _find_first_barrier_hit(prices_array, i, profit_adj, loss_adj, max_hold_days)
            labels[i] = label
            entry_dates.append(prices.index[i])
            exit_dates.append(prices.index[exit_idx])
            entry_prices.append(prices_array[i])
            exit_prices.append(prices_array[exit_idx])
            returns_pct.append((prices_array[exit_idx] - prices_array[i]) / prices_array[i])
            hold_days.append(exit_idx - i)
            barrier_hit.append(['Time', 'Profit', 'Loss'][label + 1])
            vol_adj_arr.append(vol_adj)

        # Création du DataFrame final
        self.df['Target'] = labels
        self.df['label_entry_date'] = entry_dates
        self.df['label_exit_date'] = exit_dates
        self.df['label_entry_price'] = entry_prices
        self.df['label_exit_price'] = exit_prices
        self.df['label_return'] = returns_pct
        self.df['label_hold_days'] = hold_days
        self.df['label_barrier_hit'] = barrier_hit
        self.df['vol_adjustment'] = vol_adj_arr

        return self.df

    #--- model training ---
    def RandomForest(self):
        X = self.df_features.values 
        y = self.df['Target'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        PrimaryModel =  RandomForestClassifier(
            n_estimators=100,      # 300 arbres
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
        #mise en place cross-validation

        
        #grid search
        
        #metrics
        confusion_matrix_ = confusion_matrix(y_test,PrimaryModel.predict(X_test))
        print(f"Confusion matrix : \n {confusion_matrix_}")
        classification_report_ = classification_report(y_test,PrimaryModel.predict(X_test))
        print(f"Classification report: {classification_report_}")
        #roc auc
        #roc_auc_score(y_test, PrimaryModel.predict_proba(X_test)[:, 1])
        return

# --- meta featuring --- 
def getEntropy():
    return

# --- meta labelling ---
def metaLabeling():
    return


# --- meta model ---


def main():
    ms = MomentumStrategy()
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekHigh()
    ms.getPct52WeekLow()
    ms.get12MonthPriceMomentum()
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