#package importation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yfinance as yf
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
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
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
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
        # Aligner correctement index et timestamps
        self.timestamps = pd.Series(timestamps, index=timestamps)
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.n_samples = len(labels)
        self.df = pd.DataFrame(features, index=timestamps)
        self.df['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        if label_endtimes is None:
            label_endtimes = self.timestamps
        molecules = label_endtimes.index
        all_ranges = [(start, label_endtimes[start]) for start in molecules]

        # Créer un DatetimeIndex unique pour toutes les périodes
        all_times = pd.date_range(self.timestamps.min(), self.timestamps.max(), freq='D')
        indicator = np.zeros((len(molecules), len(all_times)), dtype=np.uint8)
        time_pos = {dt: idx for idx, dt in enumerate(all_times)}

        for sample_idx, (start, end) in enumerate(all_ranges):
            if pd.isna(start) or pd.isna(end):
                continue
            rng = pd.date_range(start, end, freq='D')
            valid_idx = [time_pos[dt] for dt in rng if dt in time_pos]
            if valid_idx:
                indicator[sample_idx, valid_idx] = 1

        # S'assurer qu'aucune ligne n'est vide
        indicator[indicator.sum(axis=1) == 0, 0] = 1
        return pd.DataFrame(indicator, index=molecules, columns=all_times)

    def getAverageUniqueness(self, indicator_matrix):
        timestamp_usage_count = indicator_matrix.sum(axis=0).values
        mask = indicator_matrix.values.astype(bool)
        uniqueness_matrix = np.divide(
            mask, 
            timestamp_usage_count,
            out=np.zeros_like(mask, dtype=float),
            where=timestamp_usage_count > 0
        )
        avg_uniqueness = uniqueness_matrix.sum(axis=1) / (mask.sum(axis=1) + 1e-10)
        return pd.Series(avg_uniqueness, index=indicator_matrix.index)

    def getRarity(self):
        returns = self.df['labels']
        abs_returns = returns.abs()
        if abs_returns.sum() == 0:
            return pd.Series(np.ones(len(returns))/len(returns), index=returns.index)
        return abs_returns / abs_returns.sum()

    def getSequentialBootstrap(self, indicator_matrix, sample_length=None, random_state=42, n_simulations=100):
        np.random.seed(random_state)
        n_samples = indicator_matrix.shape[0]
        if sample_length is None:
            sample_length = n_samples
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        probabilities = avg_uniqueness / avg_uniqueness.sum()

        all_choices = np.random.choice(
            n_samples,
            size=n_simulations * sample_length,
            replace=True,
            p=probabilities.values
        ).reshape(n_simulations, sample_length)

        counts = np.bincount(all_choices.ravel(), minlength=n_samples)
        sample_weights = pd.Series(counts, index=indicator_matrix.index)
        sample_weights /= sample_weights.sum() if sample_weights.sum() > 0 else 1
        return sample_weights

    def getRecency(self, decay=0.01):
        time_delta = (self.timestamps.max() - self.timestamps).dt.days
        weights = np.exp(-decay * time_delta)
        return pd.Series(weights, index=self.timestamps.index) / weights.sum()

    def getSampleWeight(self, decay=0.01):
        indicator_matrix = self.getIndMatrix(self.timestamps)
        rarity_weights = self.getRarity()
        recency_weights = self.getRecency(decay)
        sequential_weights = self.getSequentialBootstrap(indicator_matrix)

        common_index = rarity_weights.index.intersection(recency_weights.index).intersection(sequential_weights.index)
        combined = (
            rarity_weights.loc[common_index].fillna(0) *
            recency_weights.loc[common_index].fillna(0) *
            sequential_weights.loc[common_index].fillna(0)
        )
        combined /= combined.sum() if combined.sum() > 0 else 1
        return combined

class MomentumStrategy():
    def __init__(self):
        #list of yahoo finance ticker --> 
        self.ticker = "TSLA"
        self.PERIOD = "20y"
        self.INTERVAL = "1d"
        self.SHIFT = 5
        self.lags = [12]
        self.df = self.getDataLoad()
        self.meta_df = pd.DataFrame()
        self.meta_features_df = pd.DataFrame()


    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL)
        df = df.dropna()
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['return'] = df['Close'].pct_change(self.SHIFT).shift(-self.SHIFT)
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

    def getFeaturesDataSet(self):
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close','Return','Velocity'], axis=1, errors='ignore')
        return self.df_features


    #--- labels engineering ---
    def getSignalSide(self):
        # augmenter la complexité de la logique de signal
        # Exemple de logique pour déterminer le side
        # Vous pouvez adapter selon vos critères
        conditions = []
        # threshold ?
        # Signal long si momentum positif ET RSI pas en surachat
        long_signal = (self.df['PriceMomentum'] > 0) & (self.df['RSI'] < 70) & (self.df['RSI'] > 30)
        
        # Signal short si momentum négatif ET RSI pas en survente  
        short_signal = (self.df['PriceMomentum'] < 0) & (self.df['RSI'] < 70) & (self.df['RSI'] > 30)
        
        # Créer la colonne side
        self.df['side'] = 0  # Par défaut neutre
        self.df.loc[long_signal, 'side'] = 1   # Long
        self.df.loc[short_signal, 'side'] = -1  # Short
        
        return self.df['side']

    def getLabels(self, max_hold_days=10,stop_loss = 0.01, profit_target = 0.03, volatility_scaling=True):
        prices = self.df['Close']
        n = len(prices)
        
        # S'assurer qu'on a la colonne side
        if 'side' not in self.df.columns:
            self.getSignalSide()
        
        sides = self.df['side'].values
        
        # Calcul de la volatilité roulante si scaling activé
        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std().fillna(returns.std())
        
        prices_array = prices.values
        labels = np.zeros(n)
        entry_dates, exit_dates, entry_prices, exit_prices = [], [], [], []
        returns_pct, hold_days, barrier_hit, vol_adj_arr = [], [], [], []

        def _find_first_barrier_hit_with_side(prices, entry_idx, profit_target, stop_loss, max_hold, side):
            if side == 0:  # Pas de signal
                return 0, min(entry_idx + max_hold, len(prices) - 1)
                
            entry_price = prices[entry_idx]
            end_idx = min(entry_idx + max_hold, len(prices) - 1)
            
            for i in range(entry_idx + 1, end_idx + 1):
                # Calcul du return raw
                raw_ret = (prices[i] - entry_price) / entry_price
                
                # CRUCIAL: Ajuster le return selon la direction
                # Pour short: on gagne quand le prix baisse (raw_ret négatif devient positif)
                adjusted_ret = raw_ret * side
                
                # Vérifier les barrières sur le return ajusté
                if adjusted_ret >= profit_target:
                    return 1, i  # Profit hit
                elif adjusted_ret <= -stop_loss:
                    return -1, i  # Stop loss hit
                    
            return 0, end_idx  # Time barrier hit

        for i in range(n):
            side = sides[i]
            
            # Skip si pas de signal
            if side == 0:
                labels[i] = 0
                entry_dates.append(prices.index[i])
                exit_dates.append(prices.index[i])
                entry_prices.append(prices_array[i])
                exit_prices.append(prices_array[i])
                returns_pct.append(0)
                hold_days.append(0)
                barrier_hit.append('No Signal')
                vol_adj_arr.append(1.0)
                continue
                
            # Ajustement des barrières selon volatilité
            if volatility_scaling:
                vol_filled = vol.fillna(method='bfill').fillna(method='ffill')
                vol_value = float(vol_filled.iloc[i])
                vol_adj = max(vol_value / 0.02, 0.5)
                profit_adj = profit_target * vol_adj
                loss_adj = stop_loss * vol_adj
            else:
                profit_adj = profit_target
                loss_adj = stop_loss
                vol_adj = 1.0

            # Utiliser la nouvelle fonction avec side
            label, exit_idx = _find_first_barrier_hit_with_side(
                prices_array, i, profit_adj, loss_adj, max_hold_days, side
            )
            
            labels[i] = label
            entry_dates.append(prices.index[i])
            exit_dates.append(prices.index[exit_idx])
            entry_prices.append(prices_array[i])
            exit_prices.append(prices_array[exit_idx])
            
            # IMPORTANT: Return ajusté par la direction pour le stockage
            raw_return = (prices_array[exit_idx] - prices_array[i]) / prices_array[i]
            adjusted_return = raw_return * side
            returns_pct.append(adjusted_return)
            
            hold_days.append(exit_idx - i)
            barrier_hit.append(['Time', 'Profit', 'Loss'][label + 1])
            vol_adj_arr.append(vol_adj)

        # Mise à jour du DataFrame
        self.df['Target'] = labels
        self.df['label_entry_date'] = entry_dates
        self.df['label_exit_date'] = exit_dates
        self.df['label_entry_price'] = entry_prices
        self.df['label_exit_price'] = exit_prices
        self.df['label_return'] = returns_pct  # Maintenant ajusté par direction
        self.df['label_hold_days'] = hold_days
        self.df['label_barrier_hit'] = barrier_hit
        self.df['vol_adjustment'] = vol_adj_arr
        print(self.df)
        return self.df

    def getSampleWeight(self, decay=0.01):
        labels = self.df['Target']
        features = self.df_features
        timestamps = self.df.index
        sw = SampleWeights(labels, features, timestamps)
        weights = sw.getSampleWeight(decay=decay)
        self.df['SampleWeight'] = weights
        return weights

    #--- model training ---
    def PrimaryModel(self, n_splits=5):
        X = self.df_features.values 
        y = self.df['Target'].values
        self.df['Target'].to_csv('target.csv')
        sample_weights = self.df['SampleWeight'].values  # poids calculés

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #herite de la classe PurgerKfold pour faire une 
        # cross-validation temporelle avec embargo et purge
        tscv = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []
        #gris search CV
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]
        #model tuning and hyper parameter
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )

            model.fit(X_train, y_train,sample_weight=w_train)
            y_pred = model.predict(X_test)

            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))

        print(f"\nmean_accuracy : {round((np.mean(scores)*100),2)} %")
        #Metrics
        f1_score_weighted = f1_score(y_test, y_pred, average='weighted')
        print(f"f1_score_weighted : {round(f1_score_weighted*100,2)} %")
        self.meta_df= pd.DataFrame(model.predict_proba(X_test))
        print(self.meta_df)
        return self.meta_df

    # --- meta featuring --- 
    def getEntropy(self):
        probabilities = self.meta_df.values
        # Éviter log(0) en ajoutant un epsilon
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        # Calcul de l'entropie : -sum(p * log(p))
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        self.meta_features_df['prediction_entropy'] = entropy 
        return 

    def getMaxProbability(self):
        max_probs = np.max(self.meta_df.values, axis=1)
        self.meta_features_df['max_probability'] = max_probs 
        return 
    
    def getMarginConfidence(self):
        probs = self.meta_df.values
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # Plus haute - 2ème plus haute
        self.meta_features_df['margin_confidence'] = margin
        return margin
    
    def getMetaFeaturesDf(self):
        print(self.meta_features_df)
        return self.meta_features_df

    # --- meta labelling ---
    def metaLabeling(self):
        # Condition 1: Le modèle principal a prédit un signal (non neutre)
        self.meta_df = pd.DataFrame(index=self.df.index)
        model_signal = self.df['Target'] != 0
        # Condition 2: Le trade était réellement profitable
        actual_profitable = self.df['label_return'] > 0
        # Meta-label: 1 si signal ET profitable, 0 sinon
        meta_labels = (model_signal & actual_profitable).astype(int)
        self.meta_df['meta_label'] = meta_labels
        self.meta_df.dropna()
        return 
    
    def MetaModel(self):
        X = self.meta_features_df.values
        y = self.meta_df['meta_label'].values
        
        # Split temporel
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Modèle meta (plus simple que le modèle principal)
        
        meta_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        # Entraînement
        meta_model.fit(X_train, y_train)
        
        # Prédictions
        meta_predictions = meta_model.predict_proba(X_test)[:, 1]
        self.meta_df.loc[X_test.index, 'meta_prediction'] = meta_predictions
        
        # Évaluation
        
        auc_score = roc_auc_score(y_test, meta_predictions)
        print(f"Meta-model AUC: {auc_score:.3f}")
        
        return meta_model, meta_predictions


def main():
    ms = MomentumStrategy()
    ms.getRSI()
    print("RSI implemented")
    ms.PriceMomentum()
    print("PriceMomentum implemented")
    ms.getLagReturns()
    print("LagReturns implemented")
    ms.PriceAccel()
    print("PriceAccel implemented")
    ms.getPct52WeekLow()
    print("Pct52WeekLow implemented")
    ms.getVol()
    print("Vol implemented")
    ms.getMacroData()
    print("MacroData implemented")
    ms.getFeaturesDataSet()
    print("FeaturesDataSet Created")
    ms.getSignalSide()
    print("SignalSide Created")
    ms.getLabels()
    print("Labels Created")
    ms.getSampleWeight()
    print("SampleWeight Created")
    ms.PrimaryModel()
    print("PrimaryModel Finished")
    ms.getEntropy()
    print("Entropy implemented")
    ms.getMaxProbability()
    print("MaxProbability implemented")
    ms.getMarginConfidence()
    print("MarginConfidence implemented")
    ms.getMetaFeaturesDf()
    print("MetaFeaturesDf implemented")
    ms.metaLabeling()
    print("MetaLabeling implemented")
    ms.MetaModel()
    print("MetaModel implemented")
    return

if __name__ == "__main__":
    main()