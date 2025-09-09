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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0] #nb de sample grace a la taille de la df_features
        test_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size
            test_idx = np.arange(test_start, test_end)
            train_idx = np.arange(0, test_start)
            if test_end + embargo < n_samples:
                train_idx = np.concatenate([train_idx, np.arange(test_end + embargo, n_samples)])
            yield train_idx, test_idx
        return

class SampleWeights():
    def __init__(self, labels, features, timestamps):
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.timestamps = pd.Series(timestamps)
        self.n_samples = len(labels)
        self.df = pd.DataFrame(features, index=timestamps)
        self.df['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        # Crée une matrice indicatrice binaire (samples x time) indiquant les périodes utilisées par chaque échantillon
        if label_endtimes is None:
            label_endtimes = self.timestamps
        molecules = label_endtimes.index
        all_times = pd.Index([]).union(*[pd.date_range(start, end, freq='D')
                                         for start, end in zip(molecules, label_endtimes[molecules])])
        ind_matrix = pd.DataFrame(0, index=molecules, columns=all_times)
        for sample_id in molecules:
            start_time = sample_id
            end_time = label_endtimes[sample_id]
            time_range = pd.date_range(start_time, end_time, freq='D')
            ind_matrix.loc[sample_id, time_range] = 1
        return ind_matrix

    def getAverageUniqueness(self, indicator_matrix):
        # Calcule l’unicité moyenne pour chaque échantillon sur les périodes utilisées
        timestamp_usage_count = indicator_matrix.sum(axis=0)
        uniqueness = pd.Series(index=indicator_matrix.index, dtype=float)
        for sample_id in indicator_matrix.index:
            sample_usage = indicator_matrix.loc[sample_id]
            used_timestamps = sample_usage[sample_usage == 1].index
            if len(used_timestamps) == 0:
                uniqueness[sample_id] = 0
            else:
                timestamp_uniqueness = 1.0 / timestamp_usage_count[used_timestamps]
                uniqueness[sample_id] = timestamp_uniqueness.mean()
        return uniqueness

    def getRarity(self):
        # Calcule le poids de rareté basé sur l’amplitude absolue des retours
        returns = self.df['labels']
        abs_returns = returns.abs()
        return abs_returns / abs_returns.sum()

    def getSequentialBootstrap(self, indicator_matrix, sample_length=None, random_state=42):
        # Pèse chaque échantillon selon son unicité moyenne à travers des bootstraps pondérés
        np.random.seed(random_state)
        if sample_length is None:
            sample_length = indicator_matrix.shape[0]
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        probabilities = avg_uniqueness / avg_uniqueness.sum()
        n_simulations = 10000
        selection_counts = pd.Series(0, index=indicator_matrix.index)
        for _ in range(n_simulations):
            bootstrap_indices = np.random.choice(
                indicator_matrix.index,
                size=sample_length,
                replace=True,
                p=probabilities
            )
            for idx in bootstrap_indices:
                selection_counts[idx] += 1
        sample_weights = selection_counts / selection_counts.sum()
        return sample_weights

    def getRecency(self, decay=0.01):
        # Applique une décroissance exponentielle pour valoriser les périodes récentes
        time_delta = (self.timestamps.max() - self.timestamps).dt.days
        weights = np.exp(-decay * time_delta)
        return pd.Series(weights, index=self.timestamps.index) / np.sum(weights)

    def getSampleWeight(self, decay=0.01):
        # Combine toutes les méthodes (rarete, recence, bootstrap) en un poids global et normalise
        indicator_matrix = self.getIndMatrix(self.timestamps)
        rarity_weights = self.getRarity()
        recency_weights = self.getRecency(decay)
        sequential_weights = self.getSequentialBootstrap(indicator_matrix)
        common_index = rarity_weights.index.intersection(recency_weights.index).intersection(sequential_weights.index)
        combined = rarity_weights.loc[common_index] * recency_weights.loc[common_index] * sequential_weights.loc[common_index]
        return combined / combined.sum()


class MomentumStrategy():
    def __init__(self):
        self.ticker = "TSLA"
        self.PERIOD = "max"
        self.INTERVAL = "1d"
        self.SHIFT = 5
        self.lags = [12]
        self.df = self.getDataLoad()
        self.meta_df = pd.DataFrame()

    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL)
        df = df.dropna()
        Q1 = df['Close'].quantile(0.15)
        Q3 = df['Close'].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['return'] = (df['Close'].shift(-self.SHIFT) - df['Close']) / df['Close']
        df.dropna()
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
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close','Return','Velocity'], axis=1, errors='ignore')
        return self.df_features
    
    def getMacroData(self):

        import pandas_datareader.data as web

        # Télécharger DXY et VIX via yfinance
        dxy = yf.download("DX-Y.NYB", period=self.PERIOD, interval="1d")['Close']

        # Télécharger TWI via FRED
        try:
            twi = web.DataReader("DTWEXBGS", "fred")
            twi = twi.resample("D").last()
            twi = twi['DTWEXBGS']
        except:
            twi = pd.Series(index=self.df.index, data=np.nan)

        # Réindexer et forward-fill
        self.df['DXY'] = dxy.reindex(self.df.index, method='ffill')
        self.df['TWI'] = twi.reindex(self.df.index, method='ffill')

        return self.df

    #--- statisticals test ---
    def testStationarity(self):
        for col in self.df_features.columns:
            adfuller_result = adfuller(self.df_features[col].dropna())
            p_value = adfuller_result[1]
            is_stationary = p_value < 0.05
        return

    def getCorr(self):
        self.df_features.corr()
        sns.heatmap(self.df_features.corr(), annot=True)
        return

    def getFeatureImportance(self):
        return  
    
    def getFeatureSelection(self):
        return

    def getPCATest(self):
        # Données propres (sans NaN)
        X = self.df_features.dropna()
        print(f"Shape originale: {X.shape}")
        
        # Standardisation (obligatoire pour PCA)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA pour garder 95% de la variance
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Shape après PCA: {X_pca.shape}")
        print(f"Réduction: {X.shape[1]} -> {X_pca.shape[1]} features")
        print(f"Variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Top composantes
        print(f"\nTop 5 composantes (% variance):")
        for i, var_exp in enumerate(pca.explained_variance_ratio_[:5], 1):
            print(f"PC{i}: {var_exp:.3f} ({var_exp*100:.1f}%)")
        
        # Contribution des features originales aux premières composantes
        components_df = pd.DataFrame(
            pca.components_[:3].T,  # 3 premières composantes
            columns=['PC1', 'PC2', 'PC3'],
            index=X.columns
        )
        
        print(f"\nContribution des features aux 3 premières composantes:")
        for col in ['PC1', 'PC2', 'PC3']:
            print(f"\n{col} - Top contributors:")
            top_contrib = components_df[col].abs().sort_values(ascending=False).head(3)
            for feature, contrib in top_contrib.items():
                print(f"  {feature}: {contrib:.3f}")
        
        return pca, X_pca, scaler

    
    #---  labels engineering ---
    def getLabels(self, profit_target=0.05, stop_loss=0.01, max_hold_days=10, volatility_scaling=True):
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
    def getSampleWeight(self):
        
        return
    #--- model training ---
    def PrimaryModel(self, n_splits=5):
        X = self.df_features.values 
        y = self.df['Target'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #herite de la classe PurgerKfold pour faire une 
        # cross-validation temporelle avec embargo et purge
        tscv = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        #model tuning and hyper parameter
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))

        print(f"\nmean_accuracy : {round((np.mean(scores)*100),2)} %")
        
        return np.mean(scores)

    # --- meta featuring --- 
    def getEntropy():

        return

    # --- meta labelling ---
    def metaLabeling():
        # on veut 1 si le trade etait en profit et 0 sinon

        return
    def getRealPos(self):

        return
    def getRealNeg(self):
        
        return
    def getRealNeutral(self):

        return
    def getRatio(self):

        return

    # --- meta model ---
    def MetaModel(self):
        return 

def main():
    ms = MomentumStrategy()
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekLow()
    ms.getVol()
    ms.getMacroData()
    ms.getFeaturesDataSet()
    ms.getPCATest()
    ms.getFeatureImportance()
    ms.testStationarity()
    ms.getLabels()
    ms.PrimaryModel()
    return  

if __name__ == "__main__":
    main()