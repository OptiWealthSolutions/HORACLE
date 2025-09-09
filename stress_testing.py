# Importation des bibliothèques nécessaires
import numpy as np

# Fonction pour simuler le stress testing du portefeuille
def stress_test_portfolio(mean_returns, cov_matrix, optimal_weights, num_simulations=1000, shock_factor=0.2):
    num_assets = len(optimal_weights)
    simulation_results = []

    for i in range(num_simulations):
        # Génération aléatoire des chocs pour chaque actif
        shocks = np.random.normal(0, shock_factor, num_assets)
        shocked_returns = mean_returns + shocks

        # Calcul des nouvelles performances du portefeuille après choc
        portfolio_return = np.sum(shocked_returns * optimal_weights)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

        simulation_results.append((portfolio_return, portfolio_volatility))

    return simulation_results

# Fonction pour analyser les résultats du stress test
def analyze_stress_test_results(simulation_results):
    returns = [result[0] for result in simulation_results]
    volatilities = [result[1] for result in simulation_results]

    expected_return = np.mean(returns)
    expected_volatility = np.mean(volatilities)
    min_return = np.min(returns)
    max_return = np.max(returns)

    return expected_return, expected_volatility, min_return, max_return
