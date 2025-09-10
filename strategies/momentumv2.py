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

    def getDataFrameClean(self, z_threshold: float = 4.0, iqr_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Clean the DataFrame by handling outliers using both Z-score and IQR methods.
        
        Args:
            z_threshold: Threshold for Z-score outlier detection (default: 4.0)
            iqr_multiplier: Multiplier for IQR range (default: 3.0)
            
        Returns:
            Cleaned DataFrame with outliers removed
        """
        if self.df.empty:
            raise ValueError("Input DataFrame is empty")
            
        df = self.df.copy()
        
        # 1. Calculate basic statistics
        log_returns = df['log_return'].dropna()
        if len(log_returns) < 2:
            raise ValueError("Insufficient data points for cleaning")
            
        # 2. Z-score based outlier detection
        z_scores = np.abs((log_returns - log_returns.mean()) / log_returns.std())
        
        # 3. IQR based outlier detection
        Q1 = log_returns.quantile(0.25)
        Q3 = log_returns.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - iqr_multiplier * IQR
        iqr_upper = Q3 + iqr_multiplier * IQR
        
        # 4. Combine both methods (point is outlier if detected by either method)
        is_outlier = (
            (z_scores > z_threshold) |
            (log_returns < iqr_lower) |
            (log_returns > iqr_upper)
        )
        
        # 5. Print diagnostics
        n_outliers = is_outlier.sum()
        pct_outliers = n_outliers / len(log_returns) * 100
        
        print(f"\n=== DATA CLEANING REPORT ===")
        print(f"Original data points: {len(df)}")
        print(f"Outliers detected: {n_outliers} ({pct_outliers:.2f}%)")
        print(f"Data points after cleaning: {len(df) - n_outliers}")
        
        if pct_outliers > 10:  # More than 10% outliers might indicate data issues
            print("\nWARNING: High percentage of outliers detected!")
            print("Consider investigating data quality or adjusting thresholds.")
        
        # 6. Remove outliers and return
        clean_idx = ~df.index.isin(log_returns[is_outlier].index)
        df_clean = df[clean_idx].copy()
        
        # 7. Ensure we have enough data left
        if len(df_clean) < 100:  # Arbitrary minimum threshold
            raise ValueError(
                f"Too many outliers removed. Only {len(df_clean)} points remain. "
                "Adjust thresholds or check data quality."
            )
            
        return df_clean

    # --- Features Engineering (1D uniquement) --- 
    def getRSI(self):
        #fonciton rsi

        return self.df
    
    def PriceMomentum(self):
        return self.df
    
    def getLagReturns(self):
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = self.df['log_return'].shift(n)
        return self.df
    
    def PriceAccel(self):
        self.df['velocity'] = self.df['log_return']  # Rendement = vélocité
        self.df['acceleration'] = self.df['log_return'].diff()  # Diff du rendement = accélération
        return self.df
    

    def getFeaturesDataSet(self):
        """Sélection des features 1D uniquement"""
        # Colonnes à exclure
        cols_to_drop = [
            'Open', 'High', 'Low', 'Volume', 'Close',
            'log_return', 'Label',  # Variables cibles/originales
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
        vol_21d = self.df['log_return'].rolling(21).std()
        threshold = vol_21d * 0.5  # 0.5 écart-type
        
        future_return = self.df['log_return'].shift(-1)  # t+1
        
        self.df['Label'] = np.where(
            future_return > threshold, 1,      # UP
            np.where(future_return < -threshold, -1, 0)  # DOWN, NEUTRAL
        )
        
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
    #ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    
    # Dataset features
    print("\n2. Préparation dataset...")
    ms.getFeaturesDataSet()
    
    # Tests statistiques
    print("\n3. Analyses...")
    ms.testStationarity()
    ms.getCorr()
    #ms.getFeatureImportance()
    #ms.getFeatureSelection()
    
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