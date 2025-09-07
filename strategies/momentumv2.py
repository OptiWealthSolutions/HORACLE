#package importation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ta
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class MomentumStrategy():
    def __init__(self):
        self.ticker = "EURUSD=X"
        self.PERIOD = "15y"
        self.INTERVAL = "1d"
        self.SHIFT = 5
        self.lags = [1, 2, 3, 6, 9, 12]  # Lags fixes pour features 1D
        self.df = self.getDataLoad()
        self.df = self.getDataFrameClean()

    # --- Data Loading, Cleaning and processing ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL)
        df = df.dropna()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    def getDataFrameClean(self):
        """Nettoyage minimal des outliers extrêmes"""
        df = self.df.dropna()
        
        # Seulement les outliers très extrêmes (±4 sigma)
        returns_std = df['log_return'].std()
        returns_mean = df['log_return'].mean()
        
        lower_bound = returns_mean - 4 * returns_std
        upper_bound = returns_mean + 4 * returns_std
        
        df_clean = df[(df['log_return'] >= lower_bound) & (df['log_return'] <= upper_bound)]
        print(f"Données: {len(df)} -> {len(df_clean)} (après nettoyage)")
        return df_clean

    # --- Features Engineering (1D uniquement) --- 
    def getRSI(self):
        """✅ RSI corrigé - assurer que c'est une Series 1D"""
        # S'assurer que Close est une Series pandas, pas un array
        if isinstance(self.df['Close'], pd.Series):
            close_series = self.df['Close']
        else:
            close_series = pd.Series(self.df['Close'].values, index=self.df.index)
        
        self.df['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
        return self.df
    
    def PriceMomentum(self):
        """✅ Price momentum corrigé"""
        if isinstance(self.df['Close'], pd.Series):
            close_series = self.df['Close']
        else:
            close_series = pd.Series(self.df['Close'].values, index=self.df.index)
            
        self.df['PriceMomentum'] = ta.momentum.ROCIndicator(close=close_series, window=12).roc()
        return self.df
    
    def getLagReturns(self):
        """Features de lag fixes (1D)"""
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = self.df['log_return'].shift(n)
        return self.df
    
    def PriceAccel(self):
        """Features de vélocité et accélération (1D)"""
        self.df['velocity'] = self.df['log_return']  # Rendement = vélocité
        self.df['acceleration'] = self.df['log_return'].diff()  # Diff du rendement = accélération
        return self.df
    
    def getBasicPriceFeatures(self):
        """Features de prix simples (1D)"""
        # Rendements simples
        self.df['simple_return'] = self.df['Close'].pct_change()
        
        # Position relative dans la barre OHLC
        self.df['ohlc_position'] = (self.df['Close'] - self.df['Low']) / (self.df['High'] - self.df['Low'])
        
        # Gap par rapport à l'ouverture
        self.df['open_gap'] = (self.df['Open'] / self.df['Close'].shift(1)) - 1
        
        # Body de la chandelle
        self.df['candle_body'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        
        return self.df
    
    def getTechnicalIndicators1D(self):
        """Indicateurs techniques simples (1D)"""
        # S'assurer que les séries sont 1D
        close_series = pd.Series(self.df['Close'].values, index=self.df.index)
        high_series = pd.Series(self.df['High'].values, index=self.df.index)
        low_series = pd.Series(self.df['Low'].values, index=self.df.index)
        volume_series = pd.Series(self.df['Volume'].values, index=self.df.index)
        
        # MACD line seulement (pas l'histogramme)
        macd = ta.trend.MACD(close=close_series)
        self.df['MACD'] = macd.macd()
        
        # Stochastic %K
        self.df['STOCH_K'] = ta.momentum.StochasticOscillator(
            high=high_series, low=low_series, close=close_series
        ).stoch()
        
        # Williams %R
        self.df['WILLIAMS_R'] = ta.momentum.WilliamsRIndicator(
            high=high_series, low=low_series, close=close_series
        ).williams_r()
        
        # Commodity Channel Index
        self.df['CCI'] = ta.trend.CCIIndicator(
            high=high_series, low=low_series, close=close_series
        ).cci()
        
        return self.df
    
    def getSimpleMovingAverages(self):
        """MAs simples avec ratios fixes (1D)"""
        # MA courte et longue fixes
        self.df['MA_5'] = self.df['Close'].rolling(5).mean()
        self.df['MA_20'] = self.df['Close'].rolling(20).mean()
        
        # Ratios prix/MA (features 1D)
        self.df['price_to_MA5'] = self.df['Close'] / self.df['MA_5']
        self.df['price_to_MA20'] = self.df['Close'] / self.df['MA_20']
        
        # Pente de la MA (approximation simple)
        self.df['MA5_slope'] = self.df['MA_5'].diff()
        self.df['MA20_slope'] = self.df['MA_20'].diff()
        
        return self.df

    def getFeaturesDataSet(self):
        """Sélection des features 1D uniquement"""
        # Colonnes à exclure
        cols_to_drop = [
            'Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close',
            'log_return', 'Label',  # Variables cibles/originales
            'MA_5', 'MA_20'  # MAs intermédiaires (on garde les ratios)
        ]
        
        # Garder seulement les features engineered
        all_cols = set(self.df.columns)
        cols_to_keep = all_cols - set(cols_to_drop)
        
        self.df_features = self.df[list(cols_to_keep)].copy()
        
        # Supprimer les colonnes avec trop de NaN
        na_threshold = 0.3  # Max 30% de NaN
        self.df_features = self.df_features.loc[:, self.df_features.isnull().mean() < na_threshold]
        
        print(f"Features 1D sélectionnées ({len(self.df_features.columns)}): {list(self.df_features.columns)}")
        return self.df_features
    
    #--- Statistical Tests ---
    def testStationarity(self):
        """Test de stationnarité simplifié"""
        print("\n=== TESTS DE STATIONNARITÉ ===")
        stationary_count = 0
        
        for col in self.df_features.columns:
            try:
                data = self.df_features[col].dropna()
                if len(data) < 50:
                    continue
                    
                result = adfuller(data)
                p_value = result[1]
                is_stationary = p_value < 0.05
                
                status = "STATIONNAIRE" if is_stationary else "NON-STATIONNAIRE"
                print(f"{col}: p={p_value:.4f} - {status}")
                
                if is_stationary:
                    stationary_count += 1
                    
            except Exception as e:
                print(f"Erreur pour {col}: {e}")
        
        print(f"\nRésumé: {stationary_count}/{len(self.df_features.columns)} features stationnaires")
        return
    
    def getCorr(self):
        """Analyse de corrélation"""
        print("\n=== ANALYSE DE CORRÉLATION ===")
        
        # Matrice de corrélation
        corr_matrix = self.df_features.corr()
        
        # Paires très corrélées
        high_corr_pairs = []
        n_features = len(corr_matrix.columns)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        print(f"Paires hautement corrélées (>0.8): {len(high_corr_pairs)}")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} - {feat2}: {corr:.3f}")
        
        return corr_matrix
    
    def getFeatureImportance(self):
        """Calcul importance avec modèle simple"""
        print("\n=== IMPORTANCE DES FEATURES ===")
        
        # Préparer les données
        features_clean = self.df_features.dropna()
        labels_clean = self.df.loc[features_clean.index, 'Label'].dropna()
        
        # Aligner les indices
        common_idx = features_clean.index.intersection(labels_clean.index)
        X = features_clean.loc[common_idx]
        y = labels_clean.loc[common_idx]
        
        if len(X) == 0:
            print("Pas de données pour calculer l'importance")
            return None
        
        # Modèle simple pour importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_scaled, y)
        
        # DataFrame d'importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def getFeatureSelection(self):
        """Sélection basique des features"""
        print("\n=== SÉLECTION DE FEATURES ===")
        
        # 1. Supprimer features avec variance nulle
        initial_count = len(self.df_features.columns)
        
        # Variance très faible
        low_variance_cols = []
        for col in self.df_features.columns:
            if self.df_features[col].var() < 1e-10:
                low_variance_cols.append(col)
        
        self.df_features = self.df_features.drop(columns=low_variance_cols)
        
        print(f"Features supprimées (variance nulle): {len(low_variance_cols)}")
        print(f"Features restantes: {len(self.df_features.columns)}")
        
        return self.df_features

    #--- Label Engineering ---
    def getLabels(self):
        """✅ Labels corrigés - prédire le futur"""
        # Threshold adaptatif basé sur volatilité
        vol_21d = self.df['log_return'].rolling(21).std()
        threshold = vol_21d * 0.5  # 0.5 écart-type
        
        # ✅ CRITIQUE: Utiliser rendement FUTUR
        future_return = self.df['log_return'].shift(-1)  # t+1
        
        # Labels basés sur le futur
        self.df['Label'] = np.where(
            future_return > threshold, 1,      # UP
            np.where(future_return < -threshold, -1, 0)  # DOWN, NEUTRAL
        )
        
        # Stats des labels
        label_counts = self.df['Label'].value_counts().sort_index()
        print(f"\n=== DISTRIBUTION DES LABELS ===")
        for label, count in label_counts.items():
            pct = count / len(self.df) * 100
            label_name = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(label, label)
            print(f"{label_name} ({label}): {count} ({pct:.1f}%)")
        
        return self.df

    #--- Model Training ---
    def RandomForest(self):
        """Modèle Random Forest avec validation temporelle"""
        print("\n=== ENTRAÎNEMENT RANDOM FOREST ===")
        
        # Préparation données
        df_with_labels = self.df.dropna(subset=['Label'])
        
        # Aligner features et labels
        common_idx = df_with_labels.index.intersection(self.df_features.index)
        features_final = self.df_features.loc[common_idx].dropna()
        
        # Réaligner après dropna des features
        final_idx = features_final.index.intersection(df_with_labels.index)
        X = features_final.loc[final_idx].values
        y = df_with_labels.loc[final_idx, 'Label'].values
        
        print(f"Échantillons: {len(X)}")
        print(f"Features: {X.shape[1]} (toutes 1D)")
        
        if len(X) < 100:
            print("Pas assez de données pour l'entraînement")
            return None, None
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split temporel (pas de shuffle!)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Modèle Random Forest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        print(f"\n=== RÉSULTATS ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = features_final.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== TOP 10 FEATURES IMPORTANTES ===")
        print(importance_df.head(10))
        
        return model, scaler

def main():
    print("=== STRATÉGIE MOMENTUM - FEATURES 1D UNIQUEMENT ===")
    
    ms = MomentumStrategy()
    
    # Feature Engineering (toutes 1D)
    print("\n1. Feature Engineering 1D...")
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getBasicPriceFeatures()
    ms.getTechnicalIndicators1D()
    ms.getSimpleMovingAverages()
    
    # Dataset features
    print("\n2. Préparation dataset...")
    ms.getFeaturesDataSet()
    
    # Tests statistiques
    print("\n3. Analyses...")
    ms.testStationarity()
    ms.getCorr()
    ms.getFeatureImportance()
    ms.getFeatureSelection()
    
    # Labels
    print("\n4. Création labels...")
    ms.getLabels()
    
    # Modèle
    print("\n5. Entraînement...")
    model, scaler = ms.RandomForest()
    
    print("\n=== TERMINÉ ===")
    return ms, model, scaler

if __name__ == "__main__":
    ms, model, scaler = main()