import yfinance as yf
import numpy as np
import pandas as pd
from portfolio_optimization import optimize_portfolio_with_bias, get_dividends, simulate_portfolio_performance
from report_generation import generate_pdf_report
from stress_testing import stress_test_portfolio, analyze_stress_test_results

if __name__ == "__main__":
    # 🔹 Liste des actifs
    tickers = ["CAT", "CAP.PA", "CS.PA", "LMT", "TTE", "VOL3.F", "KO"]
    start_date = '2018- 06-29'
    end_date = '2024-06-02'
    capital = 1000  # Capital initial en euros


    # 🔹 Vérification et téléchargement sécurisé des données
    valid_tickers = []
    for ticker in tickers:
        try:
            test_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not test_data.empty:
                valid_tickers.append(ticker)
                print(f"✅ Données valides pour {ticker}")
            else:
                print(f"⚠️ Aucune donnée pour {ticker} (peut être délisté).")
        except Exception as e:
            print(f"❌ Erreur pour {ticker} : {e}")

    if not valid_tickers:
        print("❌ Aucun ticker valide trouvé. Vérifiez la disponibilité des actifs sur Yahoo Finance.")
        exit()

    # 🔹 Téléchargement des données pour les tickers valides
    try:
        data = yf.download(valid_tickers, start=start_date, end=end_date, group_by='ticker', threads=False)
        
        # Assurer que les données sont bien ajustées (si 'Adj Close' est absent, utiliser 'Close')
        data_dict = {}
        for ticker in valid_tickers:
            data_dict[ticker] = data[ticker]['Close']  # Utilisation des données brutes 'Close'

        # Vérification des valeurs manquantes
        for ticker in valid_tickers:
            if data_dict[ticker].isnull().values.any():
                print(f"⚠️ Données manquantes détectées pour {ticker}, remplissage en cours...")
                data_dict[ticker] = data_dict[ticker].ffill()  # Remplissage des valeurs manquantes avec ffill
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement des données : {e}")
        exit()

    # Vérification si des données ont bien été récupérées
    if not data_dict:
        print("❌ Impossible d'optimiser le portefeuille : aucune donnée récupérée.")
        exit()

    # 🔹 Normalisation des biais pour l’optimisation
    bias_weights = np.array([8, 7, 8, 6, 7, 7, 9,5])[:len(valid_tickers)]
    bias_weights = bias_weights / np.sum(bias_weights)  # Normalisation pour éviter des pondérations excessives

    # 🔹 Calcul des rendements journaliers et création d'un DataFrame avec un index de dates
    returns_data = pd.DataFrame({ticker: data_dict[ticker].pct_change().dropna() for ticker in valid_tickers})
    
    # Vérification si le DataFrame a bien été créé
    print(returns_data.head())

    # 🔹 Optimisation du portefeuille avec contrainte de pondération entre 5% et 25%
    def constrained_optimization(*args, **kwargs):
        # Appeler l'optimisation avec des contraintes sur les pondérations
        optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, cov_matrix = optimize_portfolio_with_bias(*args, **kwargs)
        
        # Imposition des contraintes de pondération entre 5% et 25%
        optimal_weights = np.clip(optimal_weights, 0.05, 0.25)
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Re-normalisation des pondérations
        
        return optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, cov_matrix

    # 🔹 Optimisation du portefeuille
    try:
       optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, cov_matrix = constrained_optimization(
            valid_tickers, start_date, end_date, risk_free_rate=0.01, bias_weights=bias_weights
        )
    
    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation du portefeuille : {e}")
        exit()

    # 🔹 Récupération des dividendes
    try:
        dividends = get_dividends(valid_tickers)
        # Vérification des dividendes récupérés
        if dividends is None or not all(dividends):
            print("⚠️ Dividendes non disponibles ou vides pour certains tickers.")
            dividends = np.zeros(len(valid_tickers))  # Affecter une valeur nulle si les dividendes sont indisponibles
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des dividendes : {e}")
        dividends = np.zeros(len(valid_tickers))  # Affecter une valeur nulle par défaut

    # 🔹 Simulation de performance du portefeuille
    # 🔹 Simulation de performance du portefeuille
    try:
        allocation, total_expected_return, total_dividends = simulate_portfolio_performance(
            capital, optimal_weights, mean_returns, dividends, returns_data, end_date
        )
    except Exception as e:
        print(f"❌ Erreur lors de la simulation de performance : {e}")
        exit()

    # 🔹 Stress Testing du portefeuille
    try:
        simulation_results = stress_test_portfolio(mean_returns, cov_matrix, optimal_weights)
        expected_return, expected_volatility, min_return, max_return = analyze_stress_test_results(simulation_results)
    except Exception as e:
        print(f"❌ Erreur lors du stress testing : {e}")
        exit()

    # 🔹 Génération du rapport PDF
    output_file = 'portfolio_optimization_report_with_bias.pdf'
    try:
        generate_pdf_report(
            valid_tickers, optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, dividends, capital, 
            allocation, total_expected_return, total_dividends, output_file, 
            simulation_results, expected_return, expected_volatility, min_return, max_return
        )
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport PDF : {e}")
        exit()

    # 🔹 Affichage des résultats
    capital_final = capital + total_expected_return + total_dividends
    print(f"\n✅ Capital final après simulation : {capital_final:.2f} EUR")
    print(f"📊 Stress Test - Rendement attendu : {expected_return:.2f}% | Volatilité attendue : {expected_volatility:.2f}%")
    print(f"📉 Stress Test - Rendement minimum : {min_return:.2f}% | Rendement maximum : {max_return:.2f}%")