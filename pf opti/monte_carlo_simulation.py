import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Téléchargement des données de prix ajustés
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# Calcul des statistiques nécessaires (rendements moyens, matrice de covariance)
def calculate_statistics(data):
    daily_returns = data.pct_change().dropna()
    mean_daily_returns = daily_returns.mean()
    cov_matrix_daily = daily_returns.cov()
    mean_annual_returns = mean_daily_returns * 252
    cov_matrix_annual = cov_matrix_daily * 252
    return mean_annual_returns, cov_matrix_annual

# Calcul des performances du portefeuille
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Fonction de minimisation pour l'optimisation du portefeuille
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, bias_weights):
    adjusted_weights = weights * bias_weights
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
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.25 - x}
    ]
    bounds = tuple((0, 0.25) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x * bias_weights
    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormaliser les poids
    optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

    return optimal_weights, optimal_return

# Simulation Monte Carlo
def monte_carlo_simulation(tickers, start_date, end_date, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        bias_weights = np.random.uniform(0, 10, len(tickers))  # Générer des biais aléatoires entre 0 et 10
        optimal_weights, portfolio_return = optimize_portfolio_with_bias(
            tickers, start_date, end_date, bias_weights=bias_weights
        )
        results.append((bias_weights, optimal_weights, portfolio_return))
    
    results = sorted(results, key=lambda x: x[2], reverse=True)
    best_bias, best_weights, best_return = results[0]

    # Graphique des rendements en fonction des biais
    biases = [result[2] for result in results]
    plt.figure(figsize=(10, 6))
    plt.plot(biases, marker='o', linestyle='-', color='b')
    plt.title('Rendement en Fonction des Biais des Actifs')
    plt.xlabel('Scénario de Simulation')
    plt.ylabel('Rendement Annuel')
    plt.savefig('monte_carlo_results.png')
    plt.close()

    return best_bias, best_weights, best_return

def main():
    tickers = ['AAPL']
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    best_bias, best_weights, best_return = monte_carlo_simulation(tickers, start_date, end_date)
    print(f"Meilleur rendement: {best_return:.2f}")
    print(f"Meilleurs biais: {best_bias}")
    print(f"Meilleurs poids: {best_weights}")
    return

if __name__ == "__main__":
    main()
