
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from hrp_clustering_method import HRPOptimizer
from scorer import get_fundamental_data, calculate_fundamental_score

# === Pipeline structure ===
# 1. Calcul HRP weights classiques
# 2. Calcul des features fondamentales
# 3. Appliquer ML (régression ou XGBoost) pour prédire un multiplicateur de poids ou score
# 4. Ajuster HRP weights selon ML
# 5. Re-normaliser pour garder Σ(weights)=1
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "META"]
og_df = calculate_fundamental_score(tickers)
features_df = og_df.drop('Close', axis=1)

# On cherche à prédire le coefficient devant le poids/score de chaque asset
def labelling(features_df):
    # Score fondamental normalisé
    fundamental_scores = features_df['fundamental_score']
    norm_fundamental = (fundamental_scores - fundamental_scores.min()) / (fundamental_scores.max() - fundamental_scores.min() + 1e-9)

    # Calcul du retour futur sur 5 jours
    close_prices = features_df['Close']
    future_return = close_prices.shift(-5) / close_prices - 1
    # Normalisation min-max
    norm_return = (future_return - future_return.min()) / (future_return.max() - future_return.min() + 1e-9)

    # Combinaison linéaire des deux scores (poids égaux)
    combined = 0.5 * norm_fundamental + 0.5 * norm_return
    # Remplace les NaN (en fin de série) par 0
    combined = combined.fillna(0)
    # Normalisation pour que la somme = 1
    label = combined / combined.sum()

    # Retourne la série des labels
    return label

class MLScorer():
    def __init__(self, tickers):
        self.tickers = tickers
        self.fundamentals = get_fundamental_data(tickers)
        self.scores = calculate_fundamental_score(self.fundamentals)


def main():

    hrp_optimizer = HRPOptimizer(tickers)
    hrp_optimizer.optimize()
    hrp_optimizer.displayResults()
    hrp_optimizer.plotResults()
    
if __name__ == '__main__':
    main()