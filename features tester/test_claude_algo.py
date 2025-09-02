import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Classe pour calculer les indicateurs techniques sans dépendance externe"""
    
    @staticmethod
    def sma(series, window):
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series, window):
        return series.ewm(span=window).mean()
    
    @staticmethod
    def rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high, low, close, window=14):
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    @staticmethod
    def williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def atr(high, low, close, window=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(method="bfill")
    
    @staticmethod
    def obv(close, volume):
        shifted_close = close.shift()
        direction = np.where(close > shifted_close, 1, np.where(close < shifted_close, -1, 0))
        obv = (direction * volume).cumsum()
        return obv

class TradingClassificationModel:
    def __init__(self, symbol='SPY', lookback_period=20, prediction_horizon=5):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model = None
        self.features = None
        self.ti = TechnicalIndicators()
        
    def fetch_data(self, start_date='2020-01-01', end_date=None):
        """Récupère les données financières"""
        self.data = yf.download(self.symbol, start=start_date, end=end_date)
        print(f"Données récupérées: {len(self.data)} observations")
        return self.data
    
    def create_features(self):
        """Crée les features techniques pour le modèle"""
        df = self.data.copy()
        
        # Prix et rendements de base
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_momentum'] = df['Close'] / df['Close'].shift(10) - 1
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Moyennes mobiles
        df['sma_5'] = self.ti.sma(df['Close'], 5)
        df['sma_20'] = self.ti.sma(df['Close'], 20)
        df['sma_50'] = self.ti.sma(df['Close'], 50)
        df['ema_12'] = self.ti.ema(df['Close'], 12)
        df['ema_26'] = self.ti.ema(df['Close'], 26)
        
        # Ratios de prix/moyennes mobiles
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        df['sma5_sma20_ratio'] = df['sma_5'] / df['sma_20']
        df['ema12_ema26_ratio'] = df['ema_12'] / df['ema_26']
        
        # Indicateurs de momentum
        df['rsi'] = self.ti.rsi(df['Close'], 14)
        df['rsi_sma'] = self.ti.sma(df['rsi'], 3)
        df['stoch'] = self.ti.stochastic(df['High'], df['Low'], df['Close'], 14)
        df['williams_r'] = self.ti.williams_r(df['High'], df['Low'], df['Close'], 14)
        df['roc_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
        df['roc_5'] = (df['Close'] / df['Close'].shift(5) - 1) * 100
        
        # MACD
        macd_line, macd_signal, macd_histogram = self.ti.macd(df['Close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        df['macd_cross'] = np.where(macd_line > macd_signal, 1, -1)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(df['Close'], 20)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Volatilité et ATR
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['atr'] = self.ti.atr(df['High'], df['Low'], df['Close'], 14)
        df['atr_pct'] = df['atr'] / df['Close']
        
        # Volume
        df['volume_sma_10'] = self.ti.sma(df['Volume'], 10)
        df['volume_sma_20'] = self.ti.sma(df['Volume'], 20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_momentum'] = df['Volume'] / df['Volume'].shift(5)
        df['obv'] = self.ti.obv(df['Close'], df['Volume'])
        df['obv_sma'] = self.ti.sma(df['obv'], 10)
        
        # Price patterns
        df['doji'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'])
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'])
        
        # Features temporelles
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = (df.index.day > 25).astype(int)
        
        # Features de lag (retards)
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
        
        # Statistiques roulantes
        df['returns_mean_10'] = df['returns'].rolling(window=10).mean()
        df['returns_std_10'] = df['returns'].rolling(window=10).std()
        df['returns_mean_20'] = df['returns'].rolling(window=20).mean()
        df['returns_std_20'] = df['returns'].rolling(window=20).std()
        df['sharpe_10'] = df['returns_mean_10'] / df['returns_std_10']
        df['sharpe_20'] = df['returns_mean_20'] / df['returns_std_20']
        
        # Momentum croisé
        df['price_above_sma20'] = (df['Close'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['Close'] > df['sma_50']).astype(int)
        df['sma20_above_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Support/Resistance approximatifs
        df['high_20'] = df['High'].rolling(window=20).max()
        df['low_20'] = df['Low'].rolling(window=20).min()
        df['price_position_range'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        self.features_df = df
        return df
    
    def create_labels(self):
        """Crée les labels de classification (-2, -1, 0, 1, 2)"""
        df = self.features_df.copy()
        
        # Calcul du rendement futur sur prediction_horizon jours
        future_returns = df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
        
        # Définition des seuils pour la classification (ajustables)
        strong_sell_threshold = future_returns.quantile(0.15)  # 15% les plus faibles
        sell_threshold = future_returns.quantile(0.35)        # 35% les plus faibles
        buy_threshold = future_returns.quantile(0.65)         # 65% les plus élevés
        strong_buy_threshold = future_returns.quantile(0.85)  # 15% les plus élevés
        
        # Création des labels
        labels = np.zeros(len(future_returns))
        labels[future_returns <= strong_sell_threshold] = -2  # Vente forte
        labels[(future_returns > strong_sell_threshold) & (future_returns <= sell_threshold)] = -1  # Vente
        labels[(future_returns > sell_threshold) & (future_returns < buy_threshold)] = 0   # Hold
        labels[(future_returns >= buy_threshold) & (future_returns < strong_buy_threshold)] = 1    # Achat
        labels[future_returns >= strong_buy_threshold] = 2    # Achat fort
        
        df['target'] = labels
        df['future_returns'] = future_returns
        
        print(f"Distribution des labels:")
        print(pd.Series(labels).value_counts().sort_index())
        
        self.features_df = df
        return df
    
    def prepare_training_data(self):
        """Prépare les données pour l'entraînement"""
        # Sélection des features (exclure les colonnes non-features)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target', 'future_returns',
                       'bb_upper', 'bb_lower', 'obv', 'high_20', 'low_20']  # Exclure certaines features brutes
        
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        X = self.features_df[feature_cols].copy()
        y = self.features_df['target'].copy()
        
        # Suppression des lignes avec des valeurs manquantes
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Suppression des features avec trop de valeurs manquantes
        X = X.dropna(axis=1, thresh=len(X)*0.7)  # Garder les colonnes avec au moins 70% de données
        
        # Remplacer les valeurs infinies
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        print(f"Dimensions finales: X={X.shape}, y={y.shape}")
        print(f"Nombre de features: {len(X.columns)}")
        
        self.X = X
        self.y = y
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train_model(self, model_type='rf'):
        """Entraîne le modèle de classification"""
        X_scaled = self.scaler.fit_transform(self.X)
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        
        # Validation temporelle (Time Series Split)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(self.model, X_scaled, self.y, cv=tscv, scoring='accuracy')
        print(f"Accuracy moyenne (CV): {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Entraînement sur toutes les données
        self.model.fit(X_scaled, self.y)
        
        return self.model
    
    def evaluate_model(self):
        """Évalue le modèle"""
        X_scaled = self.scaler.transform(self.X)
        y_pred = self.model.predict(X_scaled)
        
        print("\nRapport de classification:")
        print(classification_report(self.y, y_pred, 
                                  target_names=['Strong Sell (-2)', 'Sell (-1)', 'Hold (0)', 'Buy (1)', 'Strong Buy (2)']))
        
        print("\nMatrice de confusion:")
        print(confusion_matrix(self.y, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 features les plus importantes:")
            print(importance_df.head(15))
            
            return importance_df
    
    def predict_signal(self, data=None):
        """Prédit le signal de trading pour les dernières données"""
        if data is None:
            # Utilise les dernières données disponibles
            latest_features = self.X.iloc[-1:].values
        else:
            latest_features = data
        
        latest_features_scaled = self.scaler.transform(latest_features)
        signal = self.model.predict(latest_features_scaled)[0]
        probability = self.model.predict_proba(latest_features_scaled)[0]
        
        signal_map = {-2: 'STRONG SELL', -1: 'SELL', 0: 'HOLD', 1: 'BUY', 2: 'STRONG BUY'}
        
        return {
            'signal': signal,
            'signal_name': signal_map[signal],
            'confidence': np.max(probability),
            'probabilities': {
                'Strong Sell (-2)': probability[0],
                'Sell (-1)': probability[1], 
                'Hold (0)': probability[2],
                'Buy (1)': probability[3],
                'Strong Buy (2)': probability[4]
            }
        }
    
    def backtest_strategy(self, start_test_date='2023-01-01'):
        """Backtest simple de la stratégie"""
        test_mask = self.features_df.index >= start_test_date
        
        if test_mask.sum() == 0:
            print("Pas de données pour la période de test spécifiée")
            return
            
        X_test = self.X[test_mask]
        y_test = self.y[test_mask]
        
        if len(X_test) == 0:
            print("Pas assez de données pour le backtest")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        # Calcule les rendements de la stratégie
        test_returns = self.features_df[test_mask]['future_returns'].values
        strategy_returns = []
        
        # Position sizing selon le signal
        position_sizes = {-2: -2, -1: -1, 0: 0, 1: 1, 2: 2}
        
        for pred, ret in zip(predictions, test_returns):
            if not np.isnan(ret):
                position = position_sizes[pred]
                strategy_returns.append(ret * position)
        
        strategy_returns = np.array(strategy_returns)
        market_returns = test_returns[~np.isnan(test_returns)][:len(strategy_returns)]
        
        # Check for empty arrays to avoid division by zero or other errors
        if len(strategy_returns) == 0:
            print("Aucun trade n'a été réalisé pendant la période de test (strategy_returns vide).")
            return
        
        # Calcul des métriques
        strategy_cumret = np.prod(1 + strategy_returns) - 1
        market_cumret = np.prod(1 + market_returns) - 1
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        market_sharpe = np.mean(market_returns) / np.std(market_returns) * np.sqrt(252) if np.std(market_returns) > 0 else 0
        
        print(f"\n📊 BACKTEST RESULTS (période: {start_test_date})")
        print("="*50)
        print(f"🔸 Rendement cumulé stratégie: {strategy_cumret:.2%}")
        print(f"🔸 Rendement cumulé marché: {market_cumret:.2%}")
        print(f"🔸 Alpha: {strategy_cumret - market_cumret:.2%}")
        print(f"🔸 Sharpe ratio stratégie: {strategy_sharpe:.2f}")
        print(f"🔸 Sharpe ratio marché: {market_sharpe:.2f}")
        print(f"🔸 Nombre de trades: {len(strategy_returns)}")
        print(f"🔸 Win rate: {(np.array(strategy_returns) > 0).mean():.2%}")

# Exemple d'utilisation
if __name__ == "__main__":
    print("🚀 Démarrage de l'algorithme de trading ML")
    print("="*50)
    
    # Initialisation du modèle
    trading_model = TradingClassificationModel(symbol='SPY', lookback_period=20, prediction_horizon=5)
    
    try:
        # Pipeline complet
        print("1️⃣ Récupération des données...")
        trading_model.fetch_data(start_date='2020-01-01')
        
        print("\n2️⃣ Création des features techniques...")
        trading_model.create_features()
        
        print("\n3️⃣ Création des labels de classification...")
        trading_model.create_labels()
        
        print("\n4️⃣ Préparation des données d'entraînement...")
        X, y = trading_model.prepare_training_data()
        
        print("\n5️⃣ Entraînement du modèle Random Forest...")
        trading_model.train_model(model_type='rf')
        
        print("\n6️⃣ Évaluation du modèle...")
        importance_df = trading_model.evaluate_model()
        
        print("\n7️⃣ Signal de trading actuel...")
        current_signal = trading_model.predict_signal()
        print(f"Signal: {current_signal['signal_name']} ({current_signal['signal']})")
        print(f"Confiance: {current_signal['confidence']:.2%}")
        
        print("\n8️⃣ Backtest de la stratégie...")
        trading_model.backtest_strategy(start_test_date='2023-01-01')
        
        print("\n✅ Algorithme prêt pour le trading!")
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        print("Vérifiez votre connexion internet et les dépendances installées")