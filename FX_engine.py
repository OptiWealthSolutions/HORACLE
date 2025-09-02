import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
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


# --- Data Loading, Cleaning and processing ---

def data_loader (ticker: str, period: str = "15y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    df['Daily_Returns'] = df[['Close']].pct_change
    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")
    return df

def data_cleaner (df):
    #deleting of extrems values
    quantile = df.quantile([0.1, 0.9])
    df = df[(df >= quantile[0.1]) & (df <= quantile[0.9])]
    #fillna
    df = df.fillna(method='ffill')
    return df

# --- Features Creating, Engineering, Comparaison and Selection ---
# Linearity, Normality, Homoscedasticity, Stationarity, Multicollinearity 

def SMA(df: pd.DataFrame, period) -> pd.DataFrame:
        # moving average features
        df[f'Mov_av_{period}'] = df['Close'].rolling(window=period).mean()
        return df

def RSI(df: pd.DataFrame, period) -> pd.DataFrame:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df[f'RSI_{period}'] = rsi
        return df

def LAG_RETURN(df, lags):
    for n in lags:
        df[f'RETURN_LAG_{n}'] = df['Close'].pct_change(periods=n)
    return df

def MACD(df):
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd_line
        df['MACD_signal'] = signal_line
        return df

#testing of features
def feature_test(df):
    print("Starting feature tests...")

    # 1. Linearity test: scatter plot Close vs lagged Close (lag 1) + regression line
    df['Close_lag1'] = df['Close'].shift(1)
    plt.figure(figsize=(8,6))
    plt.scatter(df['Close_lag1'], df['Close'], alpha=0.5, label='Data')
    # regression line
    df2 = df.dropna(subset=['Close', 'Close_lag1'])
    X = sm.add_constant(df2['Close_lag1'])
    model = sm.OLS(df2['Close'], X).fit()
    pred = model.predict(X)
    plt.plot(df2['Close_lag1'], pred, color='red', label='Regression line')
    plt.title('Linearity Test: Close vs Close_lag1')
    plt.xlabel('Close_lag1')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    # 2. Normality test: Shapiro-Wilk on returns + histogram + Q-Q plot
    returns = df['Close'].pct_change().dropna()
    stat, p = shapiro(returns)
    print(f"Normality Test (Shapiro-Wilk) on returns: stat={stat:.4f}, p-value={p:.4f}")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(returns, bins=50, alpha=0.7)
    plt.title('Histogram of Returns')
    plt.subplot(1,2,2)
    smg.qqplot(returns, line='s', ax=plt.gca())
    plt.title('Q-Q Plot of Returns')
    plt.tight_layout()
    plt.show()

    # 3. Homoscedasticity test: Breusch-Pagan on residuals of simple regression Close ~ Close_lag1
    residuals = model.resid
    bp_test = het_breuschpagan(residuals, model.model.exog)
    print(f"Homoscedasticity Test (Breusch-Pagan): LM stat={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
    # Residuals vs Fitted values plot
    plt.figure(figsize=(8,6))
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Homoscedasticity: Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # 4. Stationarity test: Augmented Dickey-Fuller on returns
    adf_result = adfuller(returns)
    print(f"Stationarity Test (ADF) on returns: ADF Statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
    # Plot returns time series
    plt.figure(figsize=(10,5))
    plt.plot(returns.index, returns.values)
    plt.title('Returns Time Series')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.show()

    # 5. Multicollinearity: VIF for features (numeric columns only)
    features = df.select_dtypes(include=[np.number]).drop(columns=['Daily_Returns'], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['feature'] = features.columns
    vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    print("Multicollinearity Test (VIF):")
    print(vif_data)
    # Bar plot of VIF values
    plt.figure(figsize=(10,5))
    plt.bar(vif_data['feature'], vif_data['VIF'])
    plt.xticks(rotation=45, ha='right')
    plt.title('VIF Values for Features')
    plt.ylabel('VIF')
    plt.tight_layout()
    plt.show()

# --- Labelling Engineering ---
# Mutliclass Threshold Labelling


# --- Model Selection ---
#Model : Random Forest Classifier

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
    # Example usage: load EURUSD data and run feature_test
    df = data_loader("EURUSD=X", period="5y")
    df = data_cleaner(df)
    # Add some features for multicollinearity to be meaningful
    df = SMA(df, 10)
    df = RSI(df, 14)
    df = LAG_RETURN(df, lags=[1,5])
    df = MACD(df)
    feature_test(df)