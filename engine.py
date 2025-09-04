import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import statsmodels.graphics.gofplots as smg
from sklearn.linear_model import LogisticRegression

#ML workflow :
#Collect Clean and Validate DATA --> Extract & Engineer Features --> Labelling --> 
# Decide ML model --> Cross Validation & Model Design & Hyper params --> Deploy and predict --> 

#what type of outcome ? ---> Mutliclass (-2,-1,0,1,2)
#what type of features ? ---> Technical, Macro, Quant
#what type of model ? ---> Random Forest Classifier
#labelling method --> rand_forest_labelling_threshold (cf: labelling_engineer)
#cross validation --> Time series cross validation
#hyper params --> Grid search


#What problem we want to solve ?
# We want to predict the future direction and strenght of the price's return for a week & month horizon 

tickerlist = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCAD=X","AUDUSD=X","NZDUSD=X"]

# --- Data Loading, Cleaning and processing ---


# --- Features Creating, Engineering, Comparaison and Selection ---
# Linearity, Normality, Homoscedasticity, Stationarity, Multicollinearity 


# --- Labelling Engineering ---
# Mutliclass Threshold Labelling and Triple barrier labelling
def getRandomForestClassifierLabel(df,prediction_horizon):
        # Calcul du rendement futur sur prediction_horizon jours
        future_returns = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        future_returns = future_returns.iloc[:, 0]  # prendre la première colonne si c'est un DataFrame
        # Définition des seuils pour la classification (ajustables)
        strong_sell_threshold = future_returns.quantile(0.15)  # 15% les plus faibles
        sell_threshold = future_returns.quantile(0.35)        # 35% les plus faibles
        buy_threshold = future_returns.quantile(0.65)         # 65% les plus élevés
        strong_buy_threshold = future_returns.quantile(0.85)  # 15% les plus élevés
        # Création des labels
        labels = np.zeros(len(future_returns))
        labels[future_returns <= strong_sell_threshold] = -2  # Vente forte
        labels[(future_returns > strong_sell_threshold) & (future_returns <= sell_threshold)] = -1  # Vente
        labels[(future_returns > sell_threshold) & (future_returns < buy_threshold)] = 0   # Hold
        labels[(future_returns >= buy_threshold) & (future_returns < strong_buy_threshold)] = 1    # Achat
        labels[future_returns >= strong_buy_threshold] = 2    # Achat fort
        df['target'] = labels
        df['future_returns'] = future_returns

        return df

def getTripleBarrierLabels():
    return
# --- Meta-Labelling ---
def getMetaLabel():
    return

# --- Model Selection ---
#Primary Model : Random Forest Classifier
def random_forest_classifier(df):
    X = df.drop(columns=['target'])
    Scaler = StandardScaler()
    X = Scaler.fit_transform(X)
    y = df['target']
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=5,max_depth=10)
    # Time Series Cross-Validation on training set
    tscv = TimeSeriesSplit(n_splits=6)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    print("Time Series Cross-Validation scores:", cv_scores)
    print("Mean CV score:", np.mean(cv_scores))
    # Fit on training set
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix on Test Set:")
    print(confusion_matrix(y_test, y_pred))
    print(y_pred)
    return model

# --- Meta-labelling Model ---


# --- Model Training, Testing, Validation ---
# train-test split
#cross-validation
#hyper params

# --- Model measurement --- 
#classification report
#confusion matrix

# --- BackTesting ---


# --- Main function activation ---

if __name__ == "__main__":
    

