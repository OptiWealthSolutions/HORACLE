import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Paramètres
# -----------------------------
TICKER = "AAPL"
PERIOD = "2y"
INTERVAL = "1d"
FUTURE_DAYS = 5
RSI_PERIOD = 14
Z_SCORE_THRESHOLD = 3  # Pour détection d'outliers

# -----------------------------
# 1. Charger les données
# -----------------------------
def load_data(ticker, period, interval):
    data = yf.download(tickers=ticker, period=period, interval=interval)
    data = data.dropna()
    return data

# -----------------------------
# 2. Calculer RSI
# -----------------------------
def compute_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------
# 3. Calculer rendement futur
# -----------------------------
def compute_future_return(data, days=5):
    future_return = (data['Close'].shift(-days) - data['Close']) / data['Close']
    return future_return

# -----------------------------
# 4. Nettoyer la feature et la target
# -----------------------------
def clean_data(df, feature_col, target_col):
    # Supprimer NaN
    df = df.dropna(subset=[feature_col, target_col])
    
    # Supprimer outliers via Z-score
    z_scores = np.abs(stats.zscore(df[[feature_col, target_col]]))
    df = df[(z_scores < Z_SCORE_THRESHOLD).all(axis=1)]
    
    return df

# -----------------------------
# 5. Préparer X et y
# -----------------------------
def prepare_features(df):
    X = df[['RSI']].values
    y = df['FutureReturn'].values
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# -----------------------------
# 6. Régression et évaluation
# -----------------------------
def run_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model, y_pred, r2

# -----------------------------
# 7. Visualisation
# -----------------------------
def plot_results(X, y, y_pred):
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Données réelles')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Prédiction')
    plt.xlabel('RSI (standardisé)')
    plt.ylabel(f'Rendement futur {FUTURE_DAYS} jours')
    plt.title(f'RSI vs Rendement futur {FUTURE_DAYS} jours')
    plt.legend()
    plt.show()

# -----------------------------
# 8. Pipeline complet
# -----------------------------
def main():
    data = load_data(TICKER, PERIOD, INTERVAL)
    
    data['RSI'] = compute_rsi(data, RSI_PERIOD)
    data['FutureReturn'] = compute_future_return(data, FUTURE_DAYS)
    
    data = clean_data(data, 'RSI', 'FutureReturn')
    
    X, y = prepare_features(data)
    
    model, y_pred, r2 = run_regression(X, y)
    
    print(f"Coefficient : {model.coef_[0]:.6f}")
    print(f"Intercept : {model.intercept_:.6f}")
    print(f"R² score : {r2:.6f}")
    
    plot_results(X, y, y_pred)

if __name__ == "__main__":
    main()