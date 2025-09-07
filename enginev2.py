import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class ForexMLPipeline:
    """
    Complete ML pipeline for forex trading prediction using Triple Barrier labeling
    and Meta-labeling approach.
    """
    
    def __init__(self, ticker: str, lookback_period: str = "2y"):
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.data = None
        self.features = None
        self.labels = None
        self.primary_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(f"ForexML_{ticker}")
        
    def load_data(self, interval: str = "1d") -> pd.DataFrame:
        """Load and validate data from Yahoo Finance"""
        try:
            self.logger.info(f"Loading data for {self.ticker}")
            data = yf.download(self.ticker, period=self.lookback_period, 
                             progress=False, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data available for {self.ticker}")
                
            # Clean data
            data = data.dropna()
            self.data = data.copy()
            self.logger.info(f"Loaded {len(data)} data points")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def check_stationarity(self, series: pd.Series, significance: float = 0.05) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return {
            'statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < significance,
            'critical_values': result[4]
        }
    
    def engineer_features(self) -> pd.DataFrame:
        """Create technical indicators and features"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        df = self.data.copy()
        
        # Price-based features
        df = self._add_moving_averages(df, [5, 10, 20, 50])
        df = self._add_rsi(df, period=14)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df, period=20)
        df = self._add_stochastic(df)
        
        # Momentum features (avoid lookahead bias)
        df = self._add_momentum_features(df, periods=[1, 3, 5, 10])
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Volume features (if applicable)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df = self._add_volume_features(df)
        
        # Remove OHLV columns and keep only features
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.features = df[feature_cols].copy()
        
        # Check for stationarity
        self._validate_features()
        
        return self.features.dropna()
    
    def _add_moving_averages(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            # Price relative to MA, robust if df['Close'] is a DataFrame
            df[f'close_sma_{period}_ratio'] = (
                df['Close'].iloc[:, 0] / df[f'sma_{period}']
                if isinstance(df['Close'], pd.DataFrame)
                else df['Close'] / df[f'sma_{period}']
            )
        return df
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        # Avoid division by zero and NaN
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi.fillna(50)  # neutral RSI if undefined
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['bb_upper'] = sma + (2 * std)
        df['bb_lower'] = sma - (2 * std)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_k = df['Low'].rolling(window=k_period).min()
        high_k = df['High'].rolling(window=k_period).max()
        denom = (high_k - low_k).replace(0, np.nan)
        stoch_k = 100 * ((df['Close'] - low_k) / denom)
        df['stoch_k'] = stoch_k.fillna(50)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add momentum features (returns over different periods)"""
        for period in periods:
            df[f'return_{period}d'] = df['Close'].pct_change(periods=period)
            
        # Add log returns
        df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Realized volatility (different windows)
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std()
            
        # High-Low volatility
        df['hl_volatility'] = (np.log(df['High']) - np.log(df['Low']))
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features (if volume data is meaningful)"""
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        return df
    
    def _validate_features(self):
        """Validate features for stationarity and handle issues"""
        if self.features is None:
            return
            
        self.logger.info("Checking feature quality...")
        
        # Remove features with too many NaN values
        nan_threshold = 0.5  # Remove if more than 50% NaN
        features_to_drop = []
        
        for col in self.features.columns:
            nan_ratio = self.features[col].isna().sum() / len(self.features)
            if nan_ratio > nan_threshold:
                features_to_drop.append(col)
            elif self.features[col].notna().sum() >= 50:  # Check stationarity if enough data
                try:
                    stationarity = self.check_stationarity(self.features[col])
                    if not stationarity['is_stationary']:
                        self.logger.debug(f"Non-stationary feature: {col}")
                except:
                    self.logger.debug(f"Could not test stationarity for: {col}")
        
        if features_to_drop:
            self.logger.warning(f"Dropping features with too many NaN: {features_to_drop}")
            self.features = self.features.drop(columns=features_to_drop)
    
    def create_events(self, threshold: float = 0.005) -> pd.DatetimeIndex:
        """Create events based on price movements (CUSUM filter)"""
        if self.data is None:
            raise ValueError("Data not loaded")
            
        close = self.data['Close']
        events = []
        s_pos, s_neg = 0, 0
        
        log_prices = np.log(close)
        diff = log_prices.diff().dropna()
        
        for timestamp in diff.index[1:]:
            pos = s_pos + diff.loc[timestamp]
            neg = s_neg + diff.loc[timestamp]
            s_pos, s_neg = max(0.0, pos), min(0.0, neg)
            
            if s_neg < -threshold:
                s_neg = 0
                events.append(timestamp)
            elif s_pos > threshold:
                s_pos = 0
                events.append(timestamp)
        
        return pd.DatetimeIndex(events)
    
    def apply_triple_barrier_labeling(self, pt_sl: Tuple[float, float] = (0.02, 0.02), 
                                    max_holding_period: int = 5) -> pd.Series:
        """Apply Triple Barrier labeling method"""
        if self.data is None:
            raise ValueError("Data not loaded")
            
        events = self.create_events()
        close = self.data['Close']
        pt, sl = pt_sl
        
        labels = pd.Series(dtype=int)

        for event_time in events:
            if event_time not in close.index:
                continue
                
            # Define time barrier
            end_time = event_time + pd.Timedelta(days=max_holding_period)
            if end_time > close.index[-1]:
                end_time = close.index[-1]
            
            # Ensure alignment: map end_time to available index using ffill
            price_path = close.loc[event_time:end_time].reindex(close.index, method='ffill').loc[event_time:end_time]
            entry_price = close.loc[event_time]
            
            # Calculate returns
            returns = (price_path / entry_price - 1.0)
            
            # Find first barrier hit
            label = 0  # Default: time barrier
            for timestamp, ret in returns.items():
                if timestamp == event_time:
                    continue
                if ret >= pt:
                    label = 1
                    break
                elif ret <= -sl:
                    label = -1
                    break
            
            labels.loc[event_time] = label
        
        self.logger.info(f"Created {len(labels)} triple barrier labels")
        self.logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        
        return labels.dropna()
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare aligned features and labels for training"""
        if self.features is None:
            self.engineer_features()
            
        if self.labels is None:
            self.labels = self.apply_triple_barrier_labeling()
        
        # Align features and labels (avoid lookahead bias)
        common_index = self.features.index.intersection(self.labels.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps between features and labels")
        
        aligned_features = self.features.loc[common_index].copy()
        aligned_labels = self.labels.loc[common_index].copy()
        
        # Remove any remaining NaN values
        valid_mask = ~(aligned_features.isna().any(axis=1) | aligned_labels.isna())
        
        final_features = aligned_features[valid_mask]
        final_labels = aligned_labels[valid_mask]
        
        self.logger.info(f"Training data shape: {final_features.shape}")
        self.logger.info(f"Final label distribution: {final_labels.value_counts().to_dict()}")
        
        return final_features, final_labels
    
    def train_primary_model(self, features: pd.DataFrame, labels: pd.Series, 
                          test_size: float = 0.2) -> Dict:
        """Train primary Random Forest classifier with time series split"""
        
        # Time series split (respecting temporal order)
        n_samples = len(features)
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = labels.iloc[:split_idx]
        y_test = labels.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.primary_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.primary_model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        tscv_scores = self._time_series_cv(X_train_scaled, y_train)
        
        # Test set evaluation
        y_pred = self.primary_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance: align index with features.columns (not scaled)
        feature_importance = pd.Series(
            self.primary_model.feature_importances_,
            index=features.columns
        ).sort_values(ascending=False)
        
        results = {
            'cv_scores': tscv_scores,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.logger.info(f"Primary model trained. Test accuracy: {test_accuracy:.4f}")
        
        return results
    
    def _time_series_cv(self, X: np.ndarray, y: pd.Series, n_splits: int = 5) -> List[float]:
        """Perform time series cross validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            )
            model.fit(X_train_cv, y_train_cv)
            score = model.score(X_val_cv, y_val_cv)
            scores.append(score)
        
        return scores
    
    def create_meta_features(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Create meta-features for meta-labeling"""
        if self.primary_model is None:
            raise ValueError("Primary model not trained")
        
        features_scaled = self.scaler.transform(features)
        primary_pred = self.primary_model.predict(features_scaled)
        primary_proba = self.primary_model.predict_proba(features_scaled)
        
        meta_features = pd.DataFrame(index=features.index)
        meta_features['max_proba'] = np.max(primary_proba, axis=1)
        meta_features['prediction_entropy'] = -np.sum(
            primary_proba * np.log2(primary_proba + 1e-10), axis=1
        )
        # Fill NaNs in rolling std with 0
        meta_features['pred_consistency'] = pd.Series(primary_pred, index=features.index).rolling(5).std().fillna(0)
        
        close = self.data['Close'].reindex(features.index, method='ffill')
        meta_features['recent_volatility'] = close.pct_change().rolling(20).std().fillna(0)
        meta_features['trend_strength'] = (
            close.rolling(10).mean() / close.rolling(30).mean() - 1
        ).fillna(0)
        
        return meta_features.dropna()
    
    def train_meta_model(self, meta_features: pd.DataFrame, labels: pd.Series, 
                        profit_threshold: float = 0.015) -> Dict:
        """Train meta-labeling model"""
        
        # Create binary meta-labels (profitable vs not profitable)
        events_data = pd.DataFrame({'labels': labels}, index=labels.index)
        events_data['returns'] = 0.0  # Placeholder - would need actual returns
        
        # For simplicity, use absolute label value as proxy for profitability
        meta_labels = (np.abs(labels) >= 1).astype(int)  # 1 if strong signal, 0 otherwise
        
        # Time series split
        n_samples = len(meta_features)
        split_idx = int(n_samples * 0.8)
        
        X_train = meta_features.iloc[:split_idx]
        X_test = meta_features.iloc[split_idx:]
        y_train = meta_labels.iloc[:split_idx]
        y_test = meta_labels.iloc[split_idx:]
        
        # Train meta model
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.meta_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.logger.info(f"Meta model trained. Accuracy: {accuracy:.4f}")
        
        return results
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """Make predictions using both primary and meta models"""
        if self.primary_model is None:
            raise ValueError("Models not trained")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Primary predictions
        primary_pred = self.primary_model.predict(features_scaled)
        primary_proba = self.primary_model.predict_proba(features_scaled)
        
        results = {
            'primary_predictions': primary_pred,
            'primary_probabilities': primary_proba,
            'timestamps': features.index.tolist()
        }
        
        # Meta predictions (if meta model is trained)
        if self.meta_model is not None:
            meta_features = self.create_meta_features(features, pd.Series(primary_pred, index=features.index))
            if len(meta_features) > 0:
                meta_pred = self.meta_model.predict(meta_features)
                results['meta_predictions'] = meta_pred
                results['final_signals'] = primary_pred * meta_pred  # Filter signals
        
        return results
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot top feature importances"""
        if self.primary_model is None:
            raise ValueError("Model not trained")
        
        importance = pd.Series(
            self.primary_model.feature_importances_,
            index=self.features.columns
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        importance.head(top_n).plot(kind='barh')
        plt.title(f'Top {top_n} Feature Importances - {self.ticker}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete ML pipeline"""
        try:
            # Load data
            self.load_data()
            
            # Engineer features
            self.engineer_features()
            
            # Create labels
            self.labels = self.apply_triple_barrier_labeling()
            
            # Prepare training data
            features, labels = self.prepare_training_data()
            
            # Train primary model
            primary_results = self.train_primary_model(features, labels)
            
            # Create and train meta model
            meta_features = self.create_meta_features(features, labels)
            meta_results = self.train_meta_model(meta_features, labels)
            
            # Combine results
            results = {
                'ticker': self.ticker,
                'data_points': len(self.data),
                'features_created': len(self.features.columns),
                'labels_created': len(self.labels),
                'training_samples': len(features),
                'primary_model': primary_results,
                'meta_model': meta_results
            }
            
            self.logger.info(f"Pipeline completed successfully for {self.ticker}")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {self.ticker}: {e}")
            raise

def run_forex_analysis(ticker_list: List[str]) -> Dict[str, Dict]:
    """Run analysis for multiple forex pairs"""
    results = {}
    
    for ticker in ticker_list:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")
        
        try:
            pipeline = ForexMLPipeline(ticker)
            ticker_results = pipeline.run_complete_pipeline()
            results[ticker] = ticker_results
            
            # Plot feature importance
            pipeline.plot_feature_importance()
            
            # Print summary
            print(f"\nSummary for {ticker}:")
            print(f"- Data points: {ticker_results['data_points']}")
            print(f"- Features: {ticker_results['features_created']}")
            print(f"- Training samples: {ticker_results['training_samples']}")
            print(f"- Primary model accuracy: {ticker_results['primary_model']['test_accuracy']:.4f}")
            print(f"- Meta model accuracy: {ticker_results['meta_model']['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results

# Main execution
if __name__ == "__main__":
    # Define forex pairs to analyze
    FOREX_PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X"]
    
    # Run complete analysis
    all_results = run_forex_analysis(FOREX_PAIRS)
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    successful_runs = {k: v for k, v in all_results.items() if 'error' not in v}
    failed_runs = {k: v for k, v in all_results.items() if 'error' in v}
    
    print(f"Successful runs: {len(successful_runs)}/{len(FOREX_PAIRS)}")
    
    if successful_runs:
        avg_accuracy = np.mean([
            results['primary_model']['test_accuracy'] 
            for results in successful_runs.values()
        ])
        print(f"Average primary model accuracy: {avg_accuracy:.4f}")
    
    if failed_runs:
        print(f"\nFailed runs: {list(failed_runs.keys())}")
        for ticker, error in failed_runs.items():
            print(f"- {ticker}: {error['error']}")