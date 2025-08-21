# Consolidated imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import yfinance as yf
import pandas_ta
from numpy import NaN as npNaN
from sklearn.cluster import KMeans
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Download S&P500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Replace '.' with '-' in ticker symbols
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()

# Define date range for data download
end_date = '2023-09-27'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

# Download historical stock data from yfinance
df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()

df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# Calculate Garman-Klass volatility estimator
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

# Calculate RSI indicator grouped by ticker
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

# Calculate Bollinger Bands (low, mid, high) using log1p of adjusted close price
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

# Define function to compute normalized ATR
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

# Calculate normalized ATR grouped by ticker
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

# Define function to compute normalized MACD
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

# Calculate normalized MACD grouped by ticker
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

# Calculate dollar volume in millions
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

print("DataFrame après calcul des indicateurs techniques:")
print(df.head())

# Aggregate to monthly level and filter top 150 most liquid stocks for each month
last_cols = [c for c in df.columns.unique() if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]

# Combine monthly mean dollar volume and last monthly values of other features
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()

print("Data après agrégation mensuelle:")
print(data.head())

# Calculate 5-year rolling average of dollar volume for each stock
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

# Rank stocks by dollar volume for each month and filter top 150
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

print("Data après filtrage top 150:")
print(data.head())

# Calculate monthly returns for different time horizons as features
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]  # Ajout de lag=1 pour return_1m
    for lag in lags:
        col_name = f'return_{lag}m'
        ret = (df['adj close']
               .pct_change(lag)
               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                      upper=x.quantile(1-outlier_cutoff)))
               .add(1)
               .pow(1/lag)
               .sub(1))
        df[col_name] = ret
    return df

# Apply calculate_returns function
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

print("Colonnes après calculate_returns:", data.columns.tolist())
print("Data shape après calculate_returns:", data.shape)

# Download Fama-French 5-factor data and prepare for regression
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'

# Join factor returns with stock returns
factor_data = factor_data.join(data['return_1m']).sort_index()

print("Factor data shape:", factor_data.shape)

# Filter out stocks with less than 10 months of data
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

print("Factor data after filtering:", factor_data.shape)

# Calculate rolling factor betas using RollingOLS regression
betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

print("Betas shape:", betas.shape)

# Join rolling factor betas to main feature dataframe
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas.groupby('ticker').shift()))

# Fill missing factor values with mean per ticker
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

# Drop adjusted close price and drop rows with missing values
data = data.drop('adj close', axis=1)
data = data.dropna()
print("Data info après nettoyage:")
data.info()

# Define target RSI values for cluster centroids (MOVED BEFORE CLUSTERING)
target_rsi_values = [30, 45, 55, 70]

# Initialize centroids array with zeros and set RSI column to target values
# RSI is at index 5 (0-indexed) in the feature matrix
rsi_column_index = list(data.columns).index('rsi')
initial_centroids = np.zeros((len(target_rsi_values), len(data.columns)))
initial_centroids[:, rsi_column_index] = target_rsi_values

print("Initial centroids shape:", initial_centroids.shape)
print("RSI column index:", rsi_column_index)

# Define function to assign clusters using KMeans with predefined centroids
def get_clusters(df):
    if len(df) < 4:  # Vérifier qu'il y a assez de données pour 4 clusters
        df['cluster'] = 0  # Assigner tous à un cluster par défaut
        return df
    
    try:
        kmeans = KMeans(n_clusters=4,
                       random_state=0,
                       init=initial_centroids,
                       n_init=1)  # Utiliser seulement 1 initialisation avec nos centroids
        df['cluster'] = kmeans.fit(df).labels_
    except Exception as e:
        print(f"Clustering failed: {e}")
        df['cluster'] = 0  # Assigner tous à un cluster par défaut
    
    return df

# Apply clustering for each month
data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

print("Data après clustering:")
print(data.head())
print("Distribution des clusters:")
print(data['cluster'].value_counts())

# Define function to plot clusters for visualization
def plot_clusters(data):
    if 'garman_klass_vol' not in data.columns or 'rsi' not in data.columns:
        print("Colonnes nécessaires pour le plot manquantes")
        return
        
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0['garman_klass_vol'], cluster_0['rsi'], color='red', label='cluster 0', alpha=0.6)
    plt.scatter(cluster_1['garman_klass_vol'], cluster_1['rsi'], color='green', label='cluster 1', alpha=0.6)
    plt.scatter(cluster_2['garman_klass_vol'], cluster_2['rsi'], color='blue', label='cluster 2', alpha=0.6)
    plt.scatter(cluster_3['garman_klass_vol'], cluster_3['rsi'], color='black', label='cluster 3', alpha=0.6)
    
    plt.xlabel('Garman-Klass Volatility')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

# Plot clusters for a few sample dates (to avoid too many plots)
sample_dates = data.index.get_level_values('date').unique()[:3]
for i in sample_dates:
    g = data.xs(i, level=0)
    if len(g) > 0:
        plt.figure(figsize=(10, 6))
        plt.title(f'Date {i.strftime("%Y-%m")}')
        plot_clusters(g)

# Filter stocks belonging to cluster 3 (high RSI cluster)
filtered_df = data[data['cluster']==3].copy()

# Shift index by 1 month to represent next month's portfolio
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(months=1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

# Get unique dates for portfolio formation
dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

# Create dictionary mapping date string to list of tickers for that date
for d in dates:
    tickers_for_date = filtered_df.xs(d, level=0).index.tolist()
    if len(tickers_for_date) > 0:  # Only add dates with at least one ticker
        fixed_dates[d.strftime('%Y-%m-%d')] = tickers_for_date

print(f"Number of portfolio formation dates: {len(fixed_dates)}")

# Define function to optimize portfolio weights maximizing Sharpe ratio
def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, 0.1),
                           solver='SCS')
    weights = ef.max_sharpe()
    return ef.clean_weights()

# Download fresh daily prices for all stocks in data
stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

print("New df shape:", new_df.shape)

# Calculate daily log returns
returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df = pd.DataFrame()

# Loop over each portfolio formation date to calculate portfolio returns
for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]
        optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        # Select price data for optimization period and chosen stocks
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
        optimization_df = optimization_df.dropna(axis=1)  # Remove stocks with missing data
        
        if len(optimization_df.columns) < 2:
            print(f'Not enough stocks for optimization at {start_date}')
            continue
            
        success = False
        try:
            # Try to optimize weights using max Sharpe ratio
            weights = optimize_weights(prices=optimization_df,
                                   lower_bound=round(1/(len(optimization_df.columns)*2),3))
            weights = pd.DataFrame(weights, index=pd.Series(0))
            success = True
        except Exception as opt_error:
            print(f'Max Sharpe Optimization failed for {start_date}: {opt_error}')
        
        # If optimization fails, assign equal weights
        if not success:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                     index=optimization_df.columns.tolist(),
                                     columns=pd.Series(0)).T
        
        # Calculate weighted returns for portfolio over the month
        temp_df = returns_dataframe[start_date:end_date]
        if len(temp_df) == 0:
            continue
            
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True,
                          how='inner')\
                   .reset_index().set_index(['Date', 'index']).unstack().stack()
        
        if len(temp_df) == 0:
            continue
            
        temp_df.index.names = ['date', 'ticker']
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

        # Append monthly portfolio returns to dataframe
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
    
    except Exception as e:
        print(f"Error processing {start_date}: {e}")

# Remove duplicate entries
portfolio_df = portfolio_df.drop_duplicates()

print("Portfolio df shape:", portfolio_df.shape)

if len(portfolio_df) > 0:
    # Download SPY data for benchmark comparison
    spy = yf.download(tickers='SPY',
                      start='2015-01-01',
                      end=dt.date.today())

    # Calculate SPY log returns
    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)

    # Merge portfolio returns with SPY returns
    portfolio_df = portfolio_df.merge(spy_ret,
                                      left_index=True,
                                      right_index=True,
                                      how='inner')

    print("Final portfolio df:")
    print(portfolio_df.head())

    def main():
        if len(portfolio_df) == 0:
            print("No portfolio returns to plot")
            return
            
        plt.style.use('ggplot')

        # Calculate cumulative returns for portfolio and SPY
        portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

        # Plot cumulative returns up to specified date
        portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))
        plt.title('Unsupervised Learning Trading Strategy Returns Over Time')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.ylabel('Return')
        plt.show()
        
        # Display some statistics
        print("\nStrategy Performance Summary:")
        print(f"Total Strategy Return: {portfolio_cumulative_return['Strategy Return'].iloc[-1]:.2%}")
        print(f"Total SPY Return: {portfolio_cumulative_return['SPY Buy&Hold'].iloc[-1]:.2%}")
        print(f"Strategy Volatility: {portfolio_df['Strategy Return'].std()*np.sqrt(252):.2%}")
        print(f"SPY Volatility: {portfolio_df['SPY Buy&Hold'].std()*np.sqrt(252):.2%}")

    if __name__ == "__main__":
        main()
else:
    print("No portfolio returns generated - check data and clustering results")