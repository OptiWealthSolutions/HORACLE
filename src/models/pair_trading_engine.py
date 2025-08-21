import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Liste des paires forex fortement corrélées
assets_correlated = [
    # Actions US fortement corrélées
    ("AAPL", "MSFT"),
    ("GOOGL", "META"),
    ("AMZN", "TSLA"),
    
    # Forex corrélés
    ("EURUSD=X", "GBPUSD=X"),
    ("USDJPY=X", "USDCAD=X"),
    ("AUDUSD=X", "EURUSD=X"),
    
    # ETF corrélés
    ("SPY", "QQQ"),
    ("DIA", "SPY"),
    ("GLD", "SLV"),
    
    # Cryptomonnaies corrélées
    ("BTC-USD", "ETH-USD"),
    ("BTC-USD", "BNB-USD"),
    ("ETH-USD", "BNB-USD")
]

def data_loader(ticker_1, ticker_2, duration, interval="60m"):
    """
    Télécharge les données pour deux tickers avec gestion d'erreurs améliorée
    """
    try:
        data_1 = yf.download(ticker_1, period=duration, interval=interval, progress=False)[["Close"]]
        data_2 = yf.download(ticker_2, period=duration, interval=interval, progress=False)[["Close"]]
        
        if data_1.empty or data_2.empty:
            raise ValueError(f"Pas de données disponibles pour {ticker_1} ou {ticker_2}")
        
        data_1.rename(columns={"Close": f"{ticker_1}_Close"}, inplace=True)
        data_2.rename(columns={"Close": f"{ticker_2}_Close"}, inplace=True)
        
        df = pd.concat([data_1, data_2], axis=1).dropna()
        
        if len(df) < 30:  # Minimum de données requises
            raise ValueError("Pas assez de données pour une analyse fiable")
            
        return df
        
    except Exception as e:
        print(f"Erreur lors du téléchargement des données: {e}")
        return pd.DataFrame()

def test_adf(series, significance_level=0.05):
    """
    Test ADF avec interprétation du niveau de significativité
    """
    result = adfuller(series)
    adf_stat, p_value = result[0], result[1]
    critical_values = result[4]
    
    is_stationary = p_value < significance_level
    
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }

def enhanced_cointegration_test(y_serie, x_serie):
    """
    Test de cointégration amélioré avec multiple approches
    """
    # Nettoyage des données
    df_temp = pd.concat([y_serie, x_serie], axis=1).dropna()
    if len(df_temp) < 30:
        return None
        
    y_clean = df_temp.iloc[:, 0]
    x_clean = df_temp.iloc[:, 1]
    
    # Test de stationnarité individuel
    y_adf = test_adf(y_clean)
    x_adf = test_adf(x_clean)
    
    # Méthode 1: Test d'Engle-Granger
    x_const = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, x_const).fit()
    residuals = model.resid
    resid_adf = test_adf(residuals)
    
    # Méthode 2: Test de Johansen via statsmodels
    try:
        coint_stat, coint_p_value, _ = coint(y_clean, x_clean)
    except:
        coint_p_value = np.nan
    
    # Calculs supplémentaires pour le trading
    spread = residuals
    zscore = (spread - spread.mean()) / spread.std()
    half_life = calculate_half_life(spread)
    
    return {
        'model': model,
        'residuals': residuals,
        'spread': spread,
        'zscore': zscore,
        'half_life': half_life,
        'engle_granger_p': resid_adf['p_value'],
        'johansen_p': coint_p_value,
        'r_squared': model.rsquared,
        'params': model.params,
        'y_stationary': y_adf['is_stationary'],
        'x_stationary': x_adf['is_stationary'],
        'residuals_stationary': resid_adf['is_stationary']
    }

def calculate_half_life(spread):
    """
    Calcule la demi-vie de retour à la moyenne du spread
    """
    try:
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        df_reg = pd.concat([spread_diff, spread_lag], axis=1).dropna()
        df_reg.columns = ['spread_diff', 'spread_lag']
        
        model = sm.OLS(df_reg['spread_diff'], 
                      sm.add_constant(df_reg['spread_lag'])).fit()
        
        half_life = -np.log(2) / model.params['spread_lag']
        return max(1, half_life)  # Minimum 1 jour
    except:
        return np.nan

def generate_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Génère des signaux de trading basés sur le z-score
    """
    signals = pd.Series(0, index=zscore.index)
    
    # Long signal (acheter la paire sous-évaluée)
    signals[zscore < -entry_threshold] = 1
    
    # Short signal (vendre la paire sur-évaluée)
    signals[zscore > entry_threshold] = -1
    
    # Exit signals
    signals[abs(zscore) < exit_threshold] = 0
    
    return signals

def pairs_trading_analysis(ticker1, ticker2, duration="1y", plot=True):
    """
    Analyse complète de pairs trading avec signaux
    """
    print(f"\n{'='*60}")
    print(f"ANALYSE DE PAIRS TRADING: {ticker1} vs {ticker2}")
    print(f"{'='*60}")
    
    # Chargement des données
    df = data_loader(ticker1, ticker2, duration)
    if df.empty:
        return None
    
    print(f"Période analysée: {df.index[0].strftime('%Y-%m-%d')} à {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Nombre d'observations: {len(df)}")
    
    # Test de cointégration
    results = enhanced_cointegration_test(
        df[f"{ticker1}_Close"], 
        df[f"{ticker2}_Close"]
    )
    
    if results is None:
        print("❌ Pas assez de données pour l'analyse")
        return None
    
    # Interprétation des résultats
    print(f"\n📊 RÉSULTATS DE COINTÉGRATION:")
    print(f"- Engle-Granger p-value: {results['engle_granger_p']:.4f}")
    print(f"- Johansen p-value: {results['johansen_p']:.4f}" if not np.isnan(results['johansen_p']) else "- Johansen: Non calculable")
    
    cointegrated = results['engle_granger_p'] < 0.05
    print(f"- Cointégration détectée: {'✅ OUI' if cointegrated else '❌ NON'}")
    print(f"- R² du modèle: {results['r_squared']:.4f}")
    print(f"- Demi-vie de retour: {results['half_life']:.1f} jours")
    
    # Équation de cointégration
    alpha, beta = results['params'][0], results['params'][1]
    print(f"- Équation: {ticker1} = {alpha:.4f} + {beta:.4f} * {ticker2} + résidus")
    
    # Analyse du z-score actuel
    current_zscore = results['zscore'].iloc[-1]
    print(f"\n🎯 SIGNAUX DE TRADING:")
    print(f"- Z-score actuel: {current_zscore:.2f}")
    
    if abs(current_zscore) > 2:
        direction = "Long" if current_zscore < -2 else "Short"
        pair_action = f"{direction} {ticker1} vs {ticker2}"
        print(f"- Signal: 🔥 {pair_action} (z-score > 2σ)")
    elif abs(current_zscore) > 1:
        print(f"- Signal: ⚠️ Surveillance recommandée")
    else:
        print(f"- Signal: 😴 Pas de signal (proche de la moyenne)")
    
    # Génération des signaux
    signals = generate_trading_signals(results['zscore'])
    recent_signals = signals.tail(10)
    active_signals = recent_signals[recent_signals != 0]
    
    if not active_signals.empty:
        print(f"- Signaux récents: {len(active_signals)} dans les 10 derniers jours")
    
    # Visualisation
    if plot and cointegrated:
        plot_analysis(df, results, ticker1, ticker2)
    
    # Score de qualité de la paire
    quality_score = calculate_pair_quality(results)
    print(f"\n⭐ SCORE DE QUALITÉ: {quality_score}/10")
    
    return {
        'tickers': (ticker1, ticker2),
        'cointegrated': cointegrated,
        'quality_score': quality_score,
        'current_zscore': current_zscore,
        'half_life': results['half_life'],
        'r_squared': results['r_squared'],
        'results': results
    }

def calculate_pair_quality(results):
    """
    Calcule un score de qualité pour la paire (0-10)
    """
    score = 0
    
    # Cointégration (4 points)
    if results['engle_granger_p'] < 0.01:
        score += 4
    elif results['engle_granger_p'] < 0.05:
        score += 2
    
    # R² (2 points)
    if results['r_squared'] > 0.8:
        score += 2
    elif results['r_squared'] > 0.5:
        score += 1
    
    # Demi-vie (2 points)
    if 1 <= results['half_life'] <= 30:
        score += 2
    elif results['half_life'] <= 60:
        score += 1
    
    # Stationnarité des résidus (2 points)
    if results['residuals_stationary']:
        score += 2
    
    return min(10, score)

def plot_analysis(df, results, ticker1, ticker2):
    """
    Visualisation de l'analyse de cointégration
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prix des deux séries
    ax1.plot(df.index, df[f"{ticker1}_Close"], label=ticker1, alpha=0.8)
    ax1.plot(df.index, df[f"{ticker2}_Close"], label=ticker2, alpha=0.8)
    ax1.set_title('Prix des Deux Paires Forex')
    ax1.legend()
    ax1.grid(True)
    
    # Spread (résidus)
    ax2.plot(df.index, results['spread'], color='red', alpha=0.7)
    ax2.axhline(results['spread'].mean(), color='black', linestyle='--', label='Moyenne')
    ax2.fill_between(df.index, 
                     results['spread'].mean() - results['spread'].std(),
                     results['spread'].mean() + results['spread'].std(),
                     alpha=0.2, color='gray', label='±1σ')
    ax2.set_title('Spread (Résidus de Cointégration)')
    ax2.legend()
    ax2.grid(True)
    
    # Z-score avec signaux
    ax3.plot(df.index, results['zscore'], color='blue', alpha=0.8)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(2, color='red', linestyle='--', alpha=0.7, label='Seuil d\'entrée')
    ax3.axhline(-2, color='red', linestyle='--', alpha=0.7)
    ax3.fill_between(df.index, -2, 2, alpha=0.1, color='green', label='Zone neutre')
    ax3.set_title('Z-Score avec Seuils de Trading')
    ax3.legend()
    ax3.grid(True)
    
    # Distribution du z-score
    ax4.hist(results['zscore'].dropna(), bins=50, alpha=0.7, density=True)
    ax4.axvline(results['zscore'].iloc[-1], color='red', linestyle='--', 
                label=f'Z-score actuel: {results["zscore"].iloc[-1]:.2f}')
    ax4.set_title('Distribution du Z-Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def scan_all_pairs(duration="1y", min_quality_score=6):
    """
    Scan toutes les paires et retourne les meilleures opportunités
    """
    opportunities = []
    
    print("🔍 SCAN DE TOUTES LES PAIRES FOREX...")
    print("="*60)
    
    for ticker1, ticker2 in forex_pairs_correlated:
        result = pairs_trading_analysis(ticker1, ticker2, duration, plot=False)
        
        if result and result['cointegrated'] and result['quality_score'] >= min_quality_score:
            opportunities.append(result)
    
    # Tri par score de qualité et z-score absolu
    opportunities.sort(key=lambda x: (x['quality_score'], abs(x['current_zscore'])), reverse=True)
    
    print(f"\n🎯 MEILLEURES OPPORTUNITÉS (Score ≥ {min_quality_score}):")
    print("="*80)
    
    for i, opp in enumerate(opportunities[:5], 1):
        t1, t2 = opp['tickers']
        signal_type = "Long" if opp['current_zscore'] < -1 else "Short" if opp['current_zscore'] > 1 else "Neutre"
        
        print(f"{i}. {t1} vs {t2}")
        print(f"   Score: {opp['quality_score']}/10 | Z-score: {opp['current_zscore']:.2f} | Signal: {signal_type}")
        print(f"   R²: {opp['r_squared']:.3f} | Demi-vie: {opp['half_life']:.1f}j")
        print()
    
    return opportunities


if __name__ == "__main__":
    scan_all_pairs(duration="1y", min_quality_score=6)

