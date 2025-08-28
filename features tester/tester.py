import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Parameters 
PERIOD = "15y"
INTERVAL = "1d"
SHIFT = 5

def load_data(ticker):
    """Load data and handle multi-column issues"""
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    
    # Fix multi-column DataFrame issue from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"✅ Data loaded: {data.shape}")
    print(f"📊 Columns: {list(data.columns)}")
    return data

def create_target(df, shift=SHIFT):
    """Create target variable - future return (percentage change)"""
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Future return as percentage change (better for correlation analysis)
    df['Target'] = close_series.pct_change(periods=shift).shift(-shift) * 100
    
    print(f"🎯 Target created - Future {shift}-period return (%)")
    print(f"Target stats: Mean={df['Target'].mean():.3f}%, Std={df['Target'].std():.3f}%")
    return df

def add_return_lag(df):
    """Add lagged return features (percentage returns)"""
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    for i in range(1, SHIFT+1):
        # Use percentage returns instead of raw differences
        df[f'Lag_{i}'] = close_series.pct_change(i) * 100
    
    print(f"📈 Added {SHIFT} lagged return features (percentage)")
    return df

def add_sma_features(df):
    """Add SMA-based features"""
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Calculate SMAs
    df['SMA20'] = close_series.rolling(20).mean()
    df['SMA50'] = close_series.rolling(50).mean()
    
    # Price relative to SMAs (normalized features)
    df['Price_vs_SMA20'] = (close_series / df['SMA20'] - 1) * 100
    df['Price_vs_SMA50'] = (close_series / df['SMA50'] - 1) * 100
    
    # SMA crossing signal
    df['Crossing_Signal'] = np.where(df['SMA20'] > df['SMA50'], 1, -1)
    
    # SMA momentum (slope)
    df['SMA20_Momentum'] = df['SMA20'].pct_change() * 100
    df['SMA50_Momentum'] = df['SMA50'].pct_change() * 100
    
    print("📊 Added SMA features: relative prices, crossing signals, momentum")
    return df 

def add_vol_features(df):
    """Add volatility features"""
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Rolling volatility (annualized)
    returns = close_series.pct_change()
    df['Volatility'] = returns.rolling(SHIFT).std() * np.sqrt(12) * 100
    
    # Volatility relative to long-term average
    vol_ma = df['Volatility'].rolling(52).mean()  # 52-week moving average
    df['Vol_Relative'] = df['Volatility'] / vol_ma
    
    print("📉 Added volatility features: absolute and relative volatility")
    return df

def test_correlation(df):
    """Test correlation with comprehensive analysis"""
    # Features to analyze
    features = [
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
        'Price_vs_SMA20', 'Price_vs_SMA50',
        'Crossing_Signal', 'SMA20_Momentum', 'SMA50_Momentum',
        'Volatility', 'Vol_Relative'
    ]
    
    print("\n" + "="*60)
    print("🔍 CORRELATION ANALYSIS - Features vs Future Returns")
    print("="*60)
    
    correlations = {}
    
    for feature in features:
        if feature not in df.columns:
            print(f"⚠️  Feature {feature} not found, skipping...")
            continue
            
        # Clean data
        clean_data = df[[feature, 'Target']].dropna()
        
        if len(clean_data) < 10:
            print(f"⚠️  Not enough data for {feature} ({len(clean_data)} samples)")
            continue
        
        # Calculate correlation
        try:
            corr_coef, p_value = pearsonr(clean_data[feature], clean_data['Target'])
            correlations[feature] = {
                'correlation': corr_coef,
                'p_value': p_value,
                'samples': len(clean_data)
            }
            
            # Significance indicator
            significance = "✅" if p_value < 0.05 else "❌"
            
            print(f"{feature:<20}: r={corr_coef:>6.3f} (p={p_value:.3f}) [{len(clean_data):>3} samples] {significance}")
            
        except Exception as e:
            print(f"❌ Error calculating correlation for {feature}: {e}")
    
    # Summary
    if correlations:
        print("\n📊 SUMMARY - Strongest Correlations:")
        print("-" * 50)
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), 
                           key=lambda x: abs(x[1]['correlation']), 
                           reverse=True)
        
        for feature, stats in sorted_corr[:5]:  # Top 5
            sig_text = "SIGNIFICANT" if stats['p_value'] < 0.05 else "not significant"
            print(f"{feature:<20}: {stats['correlation']:>6.3f} ({sig_text})")
    
    return correlations

def plot_best_correlations(df, correlations, top_n=3):
    """Plot the top correlations"""
    if not correlations:
        print("No correlations to plot")
        return
    
    # Get top correlations by absolute value
    sorted_corr = sorted(correlations.items(), 
                        key=lambda x: abs(x[1]['correlation']), 
                        reverse=True)
    
    fig, axes = plt.subplots(1, min(top_n, len(sorted_corr)), figsize=(15, 5))
    if top_n == 1:
        axes = [axes]
    
    for i, (feature, stats) in enumerate(sorted_corr[:top_n]):
        if i >= len(axes):
            break
            
        # Clean data for plotting
        plot_data = df[[feature, 'Target']].dropna()
        
        axes[i].scatter(plot_data[feature], plot_data['Target'], alpha=0.6, s=20)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Target (Future Return %)')
        axes[i].set_title(f'{feature}\nr = {stats["correlation"]:.3f}')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(plot_data[feature], plot_data['Target'], 1)
        p = np.poly1d(z)
        axes[i].plot(plot_data[feature], p(plot_data[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("🚀 Starting correlation analysis...")
    
    try:
        # Load and process data
        df = load_data('EURUSD=X')
        df = create_target(df)
        df = add_return_lag(df)
        df = add_sma_features(df)
        df = add_vol_features(df)
        
        print(f"\n📊 Final dataset: {df.shape}")
        print(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Analyze correlations
        correlations = test_correlation(df)
        
        # Plot top correlations
        if correlations:
            print("\n📈 Plotting top correlations...")
            plot_best_correlations(df, correlations, top_n=3)
        
        # Additional insights
        print("\n💡 INSIGHTS:")
        print("- Negative correlations suggest mean reversion")
        print("- Positive correlations suggest momentum")
        print("- p-value < 0.05 indicates statistical significance")
        
        return df, correlations
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Execute the analysis
if __name__ == "__main__":
    df, results = main()