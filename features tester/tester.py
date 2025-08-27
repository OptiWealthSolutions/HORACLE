import pandas as pd 
import numpy as np 
import yfinance as yf 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import brute
from sklearn.feature_selection import f_classif, mutual_info_classif



#params 
PERIOD = "15y"
INTERVAL = "1mo"

SMOOTHING_WINDOW = 14
LONG_WINDOW = 51
SHORT_WINDOW = 2

SHIFT = 5

N_CLASSES = 5

def adaptive_multiclass_labeling(returns, n_classes=5):
    percentiles = np.linspace(0, 100, n_classes + 1)
    thresholds = np.percentile(returns.dropna(), percentiles)
    labels = np.zeros(len(returns))
    for i in range(n_classes):
        mask = (returns >= thresholds[i]) & (returns < thresholds[i+1])
        labels[mask] = i
    return labels

#load data 
def load_data(ticker):
    data = yf.download(tickers=ticker, period=PERIOD, interval=INTERVAL)
    return data

#create target feature
def add_label(df):
    df['RETURNS'] = df['Close'].pct_change()
    df['TARGET'] = adaptive_multiclass_labeling(df['RETURNS'], n_classes=N_CLASSES)
    return df

#create sma features
def add_SMA_crossing(df):
    df[f'SMA{LONG_WINDOW}'] = df['Close'].rolling(window=LONG_WINDOW).mean()
    df[f'SMA{SHORT_WINDOW}'] = df['Close'].rolling(window=SHORT_WINDOW).mean()
    df['SIGNAL'] = np.where(
        df[f'SMA{LONG_WINDOW}'] > df[f'SMA{SHORT_WINDOW}'], 1,
        np.where(df[f'SMA{LONG_WINDOW}'] < df[f'SMA{SHORT_WINDOW}'], -1, 0)
    )
    return df

#create lag return features
def add_return_lag(df):
    for lag in range(1,SHIFT+1):
        df[f'RETURN_LAG_{lag}'] = df['Close'].diff(lag)
    return df

#create volatility feature
def add_volatility(df):
    df['VOLATILITY'] = df['Close'].rolling(window=SHIFT).std()
    return df


#testing the corr between feature and target with linear regression
# def linear_regression(df,x,y):
    subset  = df[[x,y]].dropna()

#     X = subset[[x]].values
#     y = subset[y].values

#     model = LinearRegression()
#     model.fit(X,y)

#     y_pred = model.predict(X)
#     r2 = r2_score(y,y_pred)
#     coefficients = model.coef_
#     intercept = model.intercept_
#     print(f"R2: {r2}")
    
#     print(f"Coefficients: {coefficients}")
#     print(f"Intercept: {intercept}")

#     plt.figure()
#     plt.title(f"Linear Regression {x} vs Target")
#     plt.scatter(X,y)
    plt.show()
    return r2, coefficients, intercept 

def compute_correlations(df):
    X = df.drop(columns=["TARGET"]).select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, "TARGET"]

    # Spearman correlation
    spearman_corr = X.corrwith(y, method="spearman").sort_values(ascending=False)
    print("=== Spearman Correlation with TARGET ===")
    print(spearman_corr)

    # ANOVA F-test
    f_vals, p_vals = f_classif(X, y)
    anova_df = pd.DataFrame({
        "Feature": X.columns,
        "F_val": f_vals,
        "p_val": p_vals
    }).sort_values("F_val", ascending=False)
    print("\n=== ANOVA F-test ===")
    print(anova_df)

    # Mutual Information
    mi = mutual_info_classif(X, y, discrete_features=False)
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_Info": mi
    }).sort_values("Mutual_Info", ascending=False)
    print("\n=== Mutual Information ===")
    print(mi_df)

#main function for execute all the code
def main():
    df = load_data("EURUSD=X")
    df = add_label(df)
    df = add_SMA_crossing(df)
    df = add_return_lag(df)
    df = add_volatility(df)
    compute_correlations(df)
    return  

df = main()