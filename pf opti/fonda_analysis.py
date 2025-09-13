import yfinance as yf
import numpy as np
from portfolio_optimization import optimize_portfolio_with_bias, get_dividends, simulate_portfolio_performance
from report_generation import generate_pdf_report
from stress_testing import stress_test_portfolio, analyze_stress_test_results

# Fonction pour récupérer des données fondamentales à partir de yfinance
def get_fundamental_data(tickers):
    fundamentals = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            pe_ratio = stock.info.get('trailingPE', 0)  # P/E ratio
            dividend_yield = stock.info.get('dividendYield', 0)  # Dividend yield
            roe = stock.info.get('returnOnEquity', 0)  # ROE
            eps_growth = stock.info.get('earningsGrowth', 0)  # Growth in EPS

            fundamentals[ticker] = {
                'PE': pe_ratio,
                'Dividend Yield': dividend_yield,
                'ROE': roe,
                'EPS Growth': eps_growth
            }
        except KeyError:
            print(f"Data not available for {ticker}")
    return fundamentals

# Fonction pour calculer une note basée sur les données fondamentales
def calculate_fundamental_score(fundamentals):
    scores = {}
    for ticker, metrics in fundamentals.items():
        # Exemples de pondérations
        pe_score = 1 / (metrics['PE'] + 1) if metrics['PE'] > 0 else 0  # Un P/E plus bas est mieux
        dividend_score = metrics['Dividend Yield']  # Un rendement plus élevé est mieux
        roe_score = metrics['ROE'] if metrics['ROE'] > 0 else 0  # Un ROE plus élevé est mieux
        growth_score = metrics['EPS Growth'] if metrics['EPS Growth'] > 0 else 0  # Une croissance positive est mieux

        # Combinaison pondérée des scores
        total_score = (0.4 * pe_score) + (0.2 * dividend_score) + (0.2 * roe_score) + (0.2 * growth_score)
        scores[ticker] = total_score

    # Normalisation des scores pour que la somme des pondérations soit égale à 1
    total_sum = sum(scores.values())
    normalized_scores = {ticker: score / total_sum for ticker, score in scores.items()}

    return normalized_scores

# Mise à jour des pondérations en fonction de l'analyse fondamentale
def update_weights_with_fundamentals(tickers, bias_weights):
    fundamentals = get_fundamental_data(tickers)
    fundamental_scores = calculate_fundamental_score(fundamentals)

    # Ajuster les biais avec les scores fondamentaux
    adjusted_weights = np.array([bias_weights[i] * fundamental_scores[ticker] for i, ticker in enumerate(tickers)])
    return adjusted_weights / np.sum(adjusted_weights)  # Renormaliser les poids

# Exemple d'utilisation de la fonction mise à jour dans l'optimisation
if __name__ == "__main__":
    tickers = ["RMS.PA", "TTE.PA", "AI.PA","MC.PA",'TLSA', "AAPL","AMZN","MSFT","AXP","AIR.PA","SU.PA"]
    start_date = '2014-06-29'
    end_date = '2024-06-29'
    capital = 1000  # Capital initial en euros

    # Création d'un biais pour chaque entreprise (à ajuster avec l'analyse fondamentale)
    bias_weights = np.array([2, 6, 4, 0, 5, 2, 9, 8, 9, 0, 9])

    # Mise à jour des poids avec l'analyse fondamentale
    updated_weights = update_weights_with_fundamentals(tickers, bias_weights)

    # Optimisation du portefeuille en utilisant les poids mis à jour
    optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, cov_matrix = optimize_portfolio_with_bias(
        tickers, start_date, end_date, risk_free_rate=0.01, bias_weights=updated_weights
    )

    # Simulation de la performance du portefeuille
    dividends = get_dividends(tickers)
    allocation, total_expected_return, total_dividends = simulate_portfolio_performance(capital, optimal_weights, mean_returns, dividends)

    # Affichage des résultats
    print(f"Allocation du capital : {allocation}")
    print(f"Rendement total attendu : {total_expected_return:.2f} EUR")
    print(f"Dividendes totaux : {total_dividends:.2f} EUR")