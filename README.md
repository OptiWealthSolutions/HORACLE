# 🧠 OptiWealth Trading System

Un agent de **trading quantitatif automatique** basé sur données de marché et macroéconomiques, avec Machine Learning et exécution via MetaTrader 5 (MT5).

---

## 🚀 Objectifs du projet
- Construire un **pipeline complet** d’analyse de marché.
- Générer des **features techniques et macroéconomiques**.
- Prédire la **direction future des prix** (hausse/baisse).
- Implémenter des **stratégies de décision robustes**.
- Automatiser l’**envoi d’ordres sur MetaTrader 5**.
- Backtester et monitorer la performance avec gestion du risque.

---

## ⚙️ Installation

### 1. Cloner le repo
```bash
git clone https://github.com/ton-repo.git
cd asset_management_ows

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 📊 Fonctionnement du pipeline

1. Data Loader
	•	Télécharge prix (Yahoo, MT5).
	•	Télécharge données macro (FRED, BCE, Fed).
	•	Nettoie et resample (mensuel ou journalier).
2. Feature Engineering
	•	Indicateurs techniques (SMA, RSI, volatilité, momentum…).
	•	Indicateurs macro (taux, inflation, chômage, spreads…).
3. Machine Learning
	•	Modèles (Logistic Regression, Random Forest, XGBoost).
	•	Validation walk-forward (pas de fuite temporelle).
4. Stratégie de décision
	•	Génère signaux BUY/SELL/NEUTRAL.
	•	Position sizing en fonction des probabilités.
	•	Gestion du risque : stop-loss, take-profit, max drawdown.
5. Exécution
	•	Envoi automatique des ordres sur MetaTrader 5 via API.
6. Backtesting
	•	Comparaison avec benchmark (buy & hold).
	•	Mesures : Sharpe Ratio, Drawdown, Win Rate.
