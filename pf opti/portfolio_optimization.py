import yfinance as yf
import numpy as np
from scipy.optimize import minimize

# Téléchargement des données de prix bruts (sans ajustement automatique)
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']  # Désactivation de l'ajustement automatique
    return data

# Calcul des statistiques nécessaires (rendements moyens, matrice de covariance)
def calculate_statistics(data):
    data = data.ffill()  # Remplir les valeurs manquantes avant de calculer les rendements quotidiens
    daily_returns = data.pct_change().dropna()
    mean_daily_returns = daily_returns.mean()
    cov_matrix_daily = daily_returns.cov()
    mean_annual_returns = mean_daily_returns * 252
    cov_matrix_annual = cov_matrix_daily * 252
    return mean_annual_returns, cov_matrix_annual

# Récupération des dividendes totaux pour chaque entreprise
def get_dividends(tickers):
    dividends = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        dividend_data = stock.dividends
        total_dividends = dividend_data.sum()
        dividends[ticker] = total_dividends
    return dividends

# Calcul des performances du portefeuille
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Fonction pour limiter les pondérations à 25 % et redistribuer les excédents
def redistribute_weights(weights):
    max_weight = 0.25
    excess = np.maximum(weights - max_weight, 0)  # Trouver les pondérations au-dessus de 25 %
    available_weight = np.sum(excess)  # Calculer le total excédent
    remaining_weight = np.sum(np.minimum(weights, max_weight))  # Pondérations valides (sous le maximum)

    if remaining_weight > 0:
        # Redistribuer l'excédent aux actifs non plafonnés proportionnellement
        adjusted_weights = np.minimum(weights, max_weight)
        # Calculer la redistribution des excédents
        redistributed_weights = (remaining_weight / np.sum(np.minimum(weights, max_weight))) * available_weight
        adjusted_weights += redistributed_weights
        adjusted_weights = np.minimum(adjusted_weights, max_weight)  # S'assurer que les poids ne dépassent pas 25 %
        return adjusted_weights
    else:
        return weights

# Fonction de minimisation pour l'optimisation du portefeuille
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, bias_weights):
    adjusted_weights = weights * bias_weights
    adjusted_weights = redistribute_weights(adjusted_weights)  # Appliquer la redistribution
    adjusted_weights = adjusted_weights / np.sum(adjusted_weights)  # Renormaliser les poids

    p_return, p_volatility = portfolio_performance(adjusted_weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    return -sharpe_ratio

# Optimisation du portefeuille avec contraintes et biais (indice de pondération)
def optimize_portfolio_with_bias(tickers, start_date, end_date, risk_free_rate=0.01, bias_weights=None):
    data = download_data(tickers, start_date, end_date)
    mean_returns, cov_matrix = calculate_statistics(data)

    num_assets = len(tickers)
    args = (mean_returns, cov_matrix, risk_free_rate, bias_weights)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for asset in range(num_assets))  # Permettre des poids jusqu'à 1 (avant redistribution)
    initial_weights = num_assets * [1. / num_assets,]

    result = minimize(negative_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x * bias_weights
    optimal_weights = redistribute_weights(optimal_weights)  # Redistribuer les poids si nécessaire
    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormaliser les poids
    optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

    return optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, cov_matrix

# Simulation des performances du portefeuille
def simulate_portfolio_performance(capital, optimal_weights, mean_returns, dividends, data, end_date):
    """
    Simule la performance du portefeuille.
    
    Parameters:
    - capital (float): Capital initial en euros.
    - optimal_weights (array): Pondérations optimales des actifs.
    - mean_returns (array): Rendements annuels moyens des actifs.
    - dividends (dict): Dividendes annuels totaux par actif.
    - data (DataFrame): Données de prix bruts pour chaque action.
    - end_date (str): Date de fin de la simulation.
    
    Returns:
    - allocation (array): Répartition du capital entre les actifs.
    - total_expected_return (float): Gain total attendu en EUR.
    - total_dividends (float): Revenus des dividendes en EUR.
    """

    # Allocation du capital selon les pondérations optimisées
    allocation = capital * optimal_weights
    
    # Vérification des allocations négatives (erreur de calcul)
    if np.any(allocation < 0):
        raise ValueError("❌ Erreur : certaines allocations sont négatives.")

    # Calcul du rendement total attendu
    total_expected_return = np.dot(allocation, mean_returns)

    # Gestion des dividendes (éviter valeurs négatives)
    tickers = list(dividends.keys())
    dividend_per_share = np.array([dividends.get(ticker, 0) for ticker in tickers])
    
    # Vérification de la validité des prix de fin de période
    last_available_date = data.index[-1]
    price_at_end = data.loc[last_available_date, tickers].values
    if np.any(price_at_end <= 0):
        raise ValueError("❌ Erreur : Un des prix des actifs est nul ou négatif.")

    # Calcul des dividendes perçus selon les actions détenues
    total_dividends = np.dot(allocation, dividend_per_share / price_at_end)

    return allocation, total_expected_return, total_dividends
