import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from scipy.optimize import brute

# Parameters 
PERIOD = "15y"
INTERVAL = "1d"
SMOOTHING_WINDOW = 14
LONG_WINDOW = 51
SHORT_WINDOW = 2
SHIFT = 5

def load_data(ticker):
    """Load financial data from Yahoo Finance and handle multi-column issues"""
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    
    # Fix multi-column DataFrame issue
    if isinstance(data.columns, pd.MultiIndex):
        # If multi-level columns, flatten them
        data.columns = data.columns.droplevel(1) if data.columns.nlevels > 1 else data.columns
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns and f'{col}' in str(data.columns):
            # Try to find the column with a different case or format
            matching_cols = [c for c in data.columns if col.lower() in str(c).lower()]
            if matching_cols:
                data[col] = data[matching_cols[0]]
    
    return data

def add_label(df):
    """Create target feature - future return (percentage change)"""
    # Ensure Close is a Series, not DataFrame
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Use forward shift for future returns and percentage change for better correlation
    df['TARGET'] = close_series.pct_change(periods=SHIFT).shift(-SHIFT) * 100  # Future return in %
    return df

def add_SMA_crossing(df):
    """Create Simple Moving Average crossing features"""
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Calculate SMAs
    df[f'SMA{LONG_WINDOW}'] = close_series.rolling(window=LONG_WINDOW).mean()
    df[f'SMA{SHORT_WINDOW}'] = close_series.rolling(window=SHORT_WINDOW).mean()
    
    # Price relative to SMAs (normalized features work better for correlation)
    df[f'PRICE_vs_SMA{LONG_WINDOW}'] = (close_series / df[f'SMA{LONG_WINDOW}'] - 1) * 100
    df[f'PRICE_vs_SMA{SHORT_WINDOW}'] = (close_series / df[f'SMA{SHORT_WINDOW}'] - 1) * 100
    
    # SMA crossing signal
    df['SMA_SIGNAL'] = np.where(
        df[f'SMA{SHORT_WINDOW}'] > df[f'SMA{LONG_WINDOW}'], 1,
        np.where(df[f'SMA{SHORT_WINDOW}'] < df[f'SMA{LONG_WINDOW}'], -1, 0)
    )
    
    # SMA slope (momentum indicator)
    df[f'SMA{LONG_WINDOW}_SLOPE'] = df[f'SMA{LONG_WINDOW}'].pct_change() * 100
    
    return df

def add_return_lag(df):
    """Create lagged return features (percentage returns)"""
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    for lag in range(1, SHIFT + 1):
        df[f'RETURN_LAG_{lag}'] = close_series.pct_change(periods=lag) * 100
    return df

def add_volatility(df):
    """Create volatility features"""
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Rolling volatility (annualized)
    returns = close_series.pct_change()
    df['VOLATILITY'] = returns.rolling(window=SHIFT).std() * np.sqrt(12) * 100
    
    # Volatility relative to historical average
    vol_rolling_mean = df['VOLATILITY'].rolling(window=52).mean()
    vol_rolling_std = df['VOLATILITY'].rolling(window=52).std()
    df['VOLATILITY_ZSCORE'] = (df['VOLATILITY'] - vol_rolling_mean) / vol_rolling_std
    
    return df

def add_additional_features(df):
    """Add more technical indicators for comprehensive analysis"""
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    high_series = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
    low_series = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
    volume_series = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
    
    # RSI-like momentum indicator
    price_change = close_series.diff()
    gains = price_change.where(price_change > 0, 0)
    losses = -price_change.where(price_change < 0, 0)
    avg_gains = gains.rolling(window=14).mean()
    avg_losses = losses.rolling(window=14).mean()
    rs = avg_gains / avg_losses
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price position within recent range
    df['PRICE_POSITION'] = ((close_series - low_series.rolling(window=14).min()) / 
                           (high_series.rolling(window=14).max() - low_series.rolling(window=14).min())) * 100
    
    # Volume-based features
    df['VOLUME_SMA'] = volume_series.rolling(window=20).mean()
    df['VOLUME_RATIO'] = volume_series / df['VOLUME_SMA']
    
    # Bollinger Band position
    sma_20 = close_series.rolling(window=20).mean()
    std_20 = close_series.rolling(window=20).std()
    df['BB_POSITION'] = (close_series - sma_20) / (2 * std_20)
    
    return df

def correlation_analysis(df, feature, target='TARGET'):
    """Comprehensive correlation analysis between feature and target"""
    # Clean data - ensure both are Series
    if feature not in df.columns or target not in df.columns:
        print(f"⚠️  Column {feature} or {target} not found in DataFrame")
        return None
        
    feature_data = df[feature]
    target_data = df[target]
    
    # Handle multi-column issues
    if isinstance(feature_data, pd.DataFrame):
        feature_data = feature_data.iloc[:, 0]
    if isinstance(target_data, pd.DataFrame):
        target_data = target_data.iloc[:, 0]
    
    subset = pd.DataFrame({feature: feature_data, target: target_data}).dropna()
    
    if len(subset) < 10:
        print(f"⚠️  Not enough data for {feature} (only {len(subset)} samples)")
        return None
    
    X = subset[feature].values
    y = subset[target].values
    
    # Calculate correlations
    try:
        pearson_corr, pearson_p = pearsonr(X, y)
        spearman_corr, spearman_p = spearmanr(X, y)
    except Exception as e:
        print(f"⚠️  Error calculating correlations for {feature}: {e}")
        return None
    
    # Linear regression
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    
    # Results
    results = {
        'feature': feature,
        'n_samples': len(subset),
        'pearson_corr': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p_value': spearman_p,
        'r2_score': r2,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'feature_mean': X.mean(),
        'feature_std': X.std(),
        'target_mean': y.mean(),
        'target_std': y.std()
    }
    
    return results, X, y, y_pred

def plot_correlation(X, y, y_pred, feature, results):
    """Create correlation plot with enhanced information"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(X, y, alpha=0.6, color='blue', s=30)
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Linear Fit (R² = {results["r2_score"]:.4f})')
    
    plt.xlabel(f'{feature}')
    plt.ylabel('TARGET (Future Return %)')
    plt.title(f'Correlation Analysis: {feature} vs Future Return\n'
              f'Pearson r = {results["pearson_corr"]:.4f} (p = {results["pearson_p_value"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with correlation info
    textstr = f'Samples: {results["n_samples"]}\n'
    textstr += f'Pearson: {results["pearson_corr"]:.4f}\n'
    textstr += f'Spearman: {results["spearman_corr"]:.4f}\n'
    textstr += f'R²: {results["r2_score"]:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

def comprehensive_feature_analysis(df, features_to_analyze=None):
    """Run comprehensive analysis on all features"""
    if features_to_analyze is None:
        # Auto-detect numeric features (excluding target and original price columns)
        excluded_cols = ['TARGET', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features_to_analyze = []
        
        for col in df.columns:
            if col not in excluded_cols:
                try:
                    # Check if column is numeric and not multi-column
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.iloc[:, 0]
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        features_to_analyze.append(col)
                except:
                    continue
    
    print("🔍 FEATURE-TARGET CORRELATION ANALYSIS")
    print("=" * 60)
    
    all_results = []
    
    for feature in features_to_analyze:
        print(f"\n📊 Analyzing: {feature}")
        print("-" * 40)
        
        result = correlation_analysis(df, feature)
        if result is None:
            continue
            
        results, X, y, y_pred = result
        all_results.append(results)
        
        # Print results
        print(f"Samples: {results['n_samples']}")
        print(f"Pearson Correlation: {results['pearson_corr']:.4f} (p-value: {results['pearson_p_value']:.4f})")
        print(f"Spearman Correlation: {results['spearman_corr']:.4f} (p-value: {results['spearman_p_value']:.4f})")
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"Linear Coefficient: {results['coefficient']:.6f}")
        
        # Significance check
        if results['pearson_p_value'] < 0.05:
            print("✅ Statistically significant correlation (p < 0.05)")
        else:
            print("❌ Not statistically significant (p >= 0.05)")
        
        # Plot
        plot_correlation(X, y, y_pred, feature, results)
    
    # Summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('pearson_corr', key=abs, ascending=False)
        
        print("\n📈 SUMMARY - FEATURES RANKED BY CORRELATION STRENGTH")
        print("=" * 80)
        
        summary_cols = ['feature', 'pearson_corr', 'pearson_p_value', 'r2_score', 'n_samples']
        print(results_df[summary_cols].to_string(index=False, float_format='%.4f'))
        
        return results_df
    
    return None

def main():
    """Main execution function"""
    print("🚀 Loading data and creating features...")
    
    try:
        # Load and prepare data
        df = load_data("EURUSD=X")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check data structure
        print(f"📋 Data types:\n{df.dtypes}")
        
        df = add_label(df)
        df = add_SMA_crossing(df)
        df = add_return_lag(df)
        df = add_volatility(df)
        df = add_additional_features(df)
        
        print(f"📊 Final data shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Check target variable
        target_stats = df['TARGET'].describe()
        print(f"🎯 Target (Future Return) Statistics:\n{target_stats}")
        
        # Define features to analyze
        features_to_analyze = [
            'RETURN_LAG_1', 'RETURN_LAG_2', 'RETURN_LAG_3', 'RETURN_LAG_4', 'RETURN_LAG_5',
            f'PRICE_vs_SMA{LONG_WINDOW}', f'PRICE_vs_SMA{SHORT_WINDOW}',
            'SMA_SIGNAL', f'SMA{LONG_WINDOW}_SLOPE',
            'VOLATILITY', 'VOLATILITY_ZSCORE',
            'RSI', 'PRICE_POSITION', 'VOLUME_RATIO', 'BB_POSITION'
        ]
        
        # Filter features that actually exist
        existing_features = [f for f in features_to_analyze if f in df.columns]
        missing_features = [f for f in features_to_analyze if f not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        print(f"📊 Analyzing {len(existing_features)} features: {existing_features}")
        
        # Run comprehensive analysis
        results_df = comprehensive_feature_analysis(df, existing_features)
        
        return df, results_df
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Execute analysis
if __name__ == "__main__":
    df, results = main()