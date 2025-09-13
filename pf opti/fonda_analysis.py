from locale import normalize
import yfinance as yf
import numpy as np
import pandas as pd

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
            eps = stock.info.get('trailingEps', 0)  # Trailing EPS
            pb_ratio = stock.info.get('priceToBook', 0)  # P/B ratio
            book_value_per_share = stock.info.get('bookValue', 0)  # Book value per share
            min_price_value = np.sqrt(22.5*eps*book_value_per_share) 

            fundamentals[ticker] = {
                'PE': pe_ratio,
                'Dividend Yield': dividend_yield,
                'ROE': roe,
                'EPS Growth': eps_growth,
                'PB ratio' : pb_ratio,
                'Book value per share' : book_value_per_share,
                'Min price value' : min_price_value
            }
        except KeyError:
            print(f"Data not available for {ticker}")
    return fundamentals

# Fonction pour calculer une note basée sur les données fondamentales
def calculate_fundamental_score(fundamentals):
    scores = {}
    for ticker, metrics in fundamentals.items():
        # logique du scorer
        pe_score = 1 / (metrics['PE'] + 1) if metrics['PE'] > 0 else 0  # Un P/E plus bas est mieux
        dividend_score = metrics['Dividend Yield']  # Un rendement plus élevé est mieux
        roe_score = metrics['ROE'] if metrics['ROE'] > 0 else 0  # Un ROE plus élevé est mieux
        growth_score = metrics['EPS Growth'] if metrics['EPS Growth'] > 0 else 0  # Une croissance positive est mieux
        pb_score = metrics['PB ratio'] if metrics['PB ratio'] > 0 else 0  # Un P/B plus bas est mieux
        book_value_score = metrics['Book value per share'] if metrics['Book value per share'] > 0 else 0  # Un Book value per share plus élevé est mieux
        # Combinaison pondérée des scores
        total_score = (0.4 * pe_score) + (0.2 * dividend_score) + (0.2 * roe_score) + (0.2 * growth_score) + (0.2 * pb_score) + (0.2 * book_value_score)
        scores[ticker] = total_score
        scores[ticker]['Min price value'] = metrics['Min price value']

    # Normalisation des scores pour que la somme des pondérations soit égale à 1
    total_sum = sum(scores.values())
    df_score = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    df_score['norm_score'] = (df_score['Score'] - min(df_score['Score'])) / (max(df_score['Score']) - min(df_score['Score']))
    df_score['min_price_value'] = fundamentals[ticker]['Min price value']
    print(df_score)
    return 

# Mise à jour des pondérations en fonction de l'analyse fondamentale
def update_weights_with_fundamentals(tickers):
    fundamentals = get_fundamental_data(tickers)
    fundamental_scores = calculate_fundamental_score(fundamentals)

    # Ajuster les biais avec les scores fondamentaux
    adjusted_weights = np.array([fundamental_scores[ticker] for ticker in tickers])
    return adjusted_weights  # Renormaliser les poids

    
# Exemple d'utilisation de la fonction mise à jour dans l'optimisation
if __name__ == "__main__":
    tickers = [      # Gold Futurestickers = [
    "HM-B.ST",     # H&M (Hennes & Mauritz, Stockholm)
    "VOLV-B.ST",   # Volvo
    "CS.PA",       # AXA
    "RHM.DE",      # Rheinmetall
    "AM.PA",       # Dassault Aviation (ou Airbus si c'est AM : vérifier ton intention)
    "CDI.PA",      # Christian Dior
    "HO.PA",       # Thales (HO sur Euronext Paris)
    "CAP.PA",      # Capgemini
    ]

    print(update_weights_with_fundamentals(tickers))