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
## import pandas_ta  # Removed due to ImportError; technical indicators are calculated manually below
npNaN = np.nan
from sklearn.cluster import KMeans
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import sys

def safe_download_sp500():
    """Télécharge la liste des actions S&P500 avec gestion d'erreur"""
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
        return sp500['Symbol'].unique().tolist()[:50]  # Limiter à 50 pour les tests
    except Exception as e:
        print(f"Erreur lors du téléchargement S&P500: {e}")
        # Liste de secours avec des tickers populaires
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
                'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'NFLX']

def download_stock_data(symbols_list, start_date, end_date):
    """Télécharge les données avec gestion d'erreur robuste"""
    print(f"Téléchargement de {len(symbols_list)} actions...")
    
    # Télécharger par lots pour éviter les timeouts
    batch_size = 10
    all_data = []
    
    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        try:
            print(f"Téléchargement batch {i//batch_size + 1}/{(len(symbols_list)//batch_size) + 1}")
            batch_data = yf.download(tickers=batch,
                                   start=start_date,
                                   end=end_date,
                                   group_by='ticker',
                                   auto_adjust=True,
                                   prepost=True,
                                   threads=True)
            
            if not batch_data.empty:
                # Restructurer les données si plusieurs tickers
                if len(batch) > 1:
                    batch_data = batch_data.stack(level=0).rename_axis(['Date', 'Ticker'])
                else:
                    batch_data['Ticker'] = batch[0]
                    batch_data = batch_data.reset_index().set_index(['Date', 'Ticker'])
                
                all_data.append(batch_data)
        except Exception as e:
            print(f"Erreur téléchargement batch {batch}: {e}")
            continue
    
    if all_data:
        df = pd.concat(all_data, axis=0)
        df.index.names = ['date', 'ticker']
        df.columns = df.columns.str.lower()
        return df
    else:
        raise ValueError("Aucune donnée téléchargée avec succès")

def calculate_technical_indicators(df):
    """Calcule les indicateurs techniques avec gestion d'erreur"""
    print("Calcul des indicateurs techniques...")
    
    try:
        # Garman-Klass volatility estimator
        df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['close'])-np.log(df['open']))**2)
    except Exception as e:
        print(f"Erreur Garman-Klass: {e}")
        df['garman_klass_vol'] = df.groupby('ticker')['close'].pct_change().rolling(20).std()

    # RSI (manual implementation)
    try:
        def manual_rsi(series, window=20):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.rolling(window=window, min_periods=window).mean()
            roll_down = down.rolling(window=window, min_periods=window).mean()
            rs = roll_up / (roll_down + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            return rsi
        df['rsi'] = df.groupby('ticker', group_keys=False)['close'].apply(lambda x: manual_rsi(x, window=20))
    except Exception as e:
        print(f"Erreur RSI: {e}")
        df['rsi'] = 50  # Valeur par défaut

    # Bollinger Bands (rolling mean and std)
    try:
        def manual_bbands(close_series, window=20):
            sma = close_series.rolling(window).mean()
            std = close_series.rolling(window).std()
            lower = sma - 2 * std
            upper = sma + 2 * std
            middle = sma
            bb = pd.DataFrame({'lower': lower, 'middle': middle, 'upper': upper}, index=close_series.index)
            return bb
        bb_data = df.groupby('ticker', group_keys=False)['close'].apply(lambda x: manual_bbands(x, window=20))
        if not bb_data.empty:
            df['bb_low'] = bb_data['lower']
            df['bb_mid'] = bb_data['middle']
            df['bb_high'] = bb_data['upper']
        else:
            df['bb_low'] = df['close'] * 0.95
            df['bb_mid'] = df['close']
            df['bb_high'] = df['close'] * 1.05
    except Exception as e:
        print(f"Erreur Bollinger Bands: {e}")
        # Valeurs par défaut
        sma20 = df.groupby('ticker')['close'].rolling(20).mean().reset_index(level=0, drop=True)
        std20 = df.groupby('ticker')['close'].rolling(20).std().reset_index(level=0, drop=True)
        df['bb_low'] = sma20 - 2 * std20
        df['bb_mid'] = sma20
        df['bb_high'] = sma20 + 2 * std20

    # ATR normalisé (manual calculation)
    def manual_atr(stock_data, window=14):
        try:
            if len(stock_data) < window:
                return pd.Series(0.02, index=stock_data.index)
            high_low = stock_data['high'] - stock_data['low']
            high_close = (stock_data['high'] - stock_data['close'].shift()).abs()
            low_close = (stock_data['low'] - stock_data['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window).mean()
            if atr.std() > 0:
                return (atr - atr.mean()) / atr.std()
            else:
                return atr.fillna(0)
        except Exception as e:
            print(f"Erreur ATR pour un stock: {e}")
            return pd.Series(0.02, index=stock_data.index)
    df['atr'] = df.groupby('ticker', group_keys=False).apply(lambda x: manual_atr(x, window=14))

    # MACD normalisé (manual calculation)
    def manual_macd(close_series):
        try:
            if len(close_series) < 26:
                return pd.Series(0, index=close_series.index)
            ema12 = close_series.ewm(span=12, adjust=False).mean()
            ema26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            if macd_line.std() > 0:
                return (macd_line - macd_line.mean()) / macd_line.std()
            else:
                return macd_line.fillna(0)
        except Exception as e:
            print(f"Erreur MACD: {e}")
            return pd.Series(0, index=close_series.index)
    df['macd'] = df.groupby('ticker', group_keys=False)['close'].apply(manual_macd)

    # Dollar volume
    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6
    
    return df

def aggregate_monthly_data(df):
    """Agrège les données au niveau mensuel"""
    print("Agrégation des données mensuelles...")
    
    last_cols = [c for c in df.columns.unique() if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    try:
        # Agrégation mensuelle
        monthly_dollar_vol = df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume')
        monthly_other = df.unstack()[last_cols].resample('M').last().stack('ticker')
        
        data = pd.concat([monthly_dollar_vol, monthly_other], axis=1).dropna()
        
        # Rolling average du dollar volume
        data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker')
                                .rolling(5*12, min_periods=12).mean().stack())
        
        # Filtrage top 150 par liquidité
        data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
        data = data[data['dollar_vol_rank'] < 50].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)  # Top 50 pour les tests
        
        return data
        
    except Exception as e:
        print(f"Erreur agrégation: {e}")
        return df.resample('M').last()

def calculate_returns(df):
    """Calcule les rendements mensuels"""
    print("Calcul des rendements...")
    
    def safe_calculate_returns(group_df):
        try:
            outlier_cutoff = 0.005
            lags = [1, 2, 3, 6, 9, 12]
            
            for lag in lags:
                if len(group_df) > lag:
                    col_name = f'return_{lag}m'
                    ret = (group_df['close'].pct_change(lag)
                          .clip(lower=group_df['close'].pct_change(lag).quantile(outlier_cutoff),
                                upper=group_df['close'].pct_change(lag).quantile(1-outlier_cutoff))
                          .add(1).pow(1/lag).sub(1))
                    group_df[col_name] = ret
                else:
                    group_df[f'return_{lag}m'] = 0
                    
            return group_df
        except Exception as e:
            print(f"Erreur calcul rendements: {e}")
            # Retours par défaut
            for lag in [1, 2, 3, 6, 9, 12]:
                group_df[f'return_{lag}m'] = 0
            return group_df
    
    if 'close' not in df.columns:
        # Si pas de colonne close, créer return_1m à partir de adj_close ou close disponible
        price_col = 'adj_close' if 'adj_close' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        df['return_1m'] = df.groupby('ticker')[price_col].pct_change()
        return df
    
    data = df.groupby('ticker', group_keys=False).apply(safe_calculate_returns)
    
    # Vérifier que return_1m existe
    if 'return_1m' not in data.columns:
        print("Création manuelle de return_1m...")
        data['return_1m'] = data.groupby('ticker')['close'].pct_change()
    
    return data.dropna()

def get_fama_french_factors():
    """Télécharge les facteurs Fama-French avec gestion d'erreur"""
    try:
        print("Téléchargement des facteurs Fama-French...")
        factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0]
        factor_data = factor_data.drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample('M').last().div(100)
        factor_data.index.name = 'date'
        return factor_data
    except Exception as e:
        print(f"Erreur téléchargement Fama-French: {e}")
        # Créer des facteurs fictifs
        dates = pd.date_range(start='2015-01-31', end='2023-12-31', freq='M')
        np.random.seed(42)
        factor_data = pd.DataFrame({
            'Mkt-RF': np.random.normal(0.01, 0.05, len(dates)),
            'SMB': np.random.normal(0, 0.03, len(dates)),
            'HML': np.random.normal(0, 0.03, len(dates)),
            'RMW': np.random.normal(0, 0.03, len(dates)),
            'CMA': np.random.normal(0, 0.03, len(dates))
        }, index=dates)
        factor_data.index.name = 'date'
        return factor_data

def calculate_factor_betas(data, factor_data):
    """Calcule les bêtas des facteurs avec gestion d'erreur"""
    print("Calcul des bêtas factoriels...")
    
    try:
        # Joindre les données
        factor_data_with_returns = factor_data.join(data['return_1m']).sort_index()
        
        # Filtrer les actions avec suffisamment de données
        observations = factor_data_with_returns.groupby('ticker').size()
        valid_stocks = observations[observations >= 10].index
        factor_data_with_returns = factor_data_with_returns[
            factor_data_with_returns.index.get_level_values('ticker').isin(valid_stocks)
        ]
        
        def safe_rolling_ols(group):
            try:
                if len(group) < 12:  # Minimum de données
                    # Retourner des bêtas par défaut
                    betas = pd.DataFrame(index=group.index, 
                                       columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
                    betas['Mkt-RF'] = 1.0  # Bêta marché par défaut
                    betas[['SMB', 'HML', 'RMW', 'CMA']] = 0.0
                    return betas
                
                y = group['return_1m'].dropna()
                X = group[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].loc[y.index]
                
                if len(y) < 6:  # Pas assez de données
                    betas = pd.DataFrame(index=group.index, 
                                       columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
                    betas['Mkt-RF'] = 1.0
                    betas[['SMB', 'HML', 'RMW', 'CMA']] = 0.0
                    return betas
                
                # Rolling OLS
                model = RollingOLS(endog=y, 
                                 exog=sm.add_constant(X),
                                 window=min(24, len(y)),
                                 min_nobs=5)
                results = model.fit(params_only=True)
                return results.params.drop('const', axis=1)
            
            except Exception as e:
                print(f"Erreur Rolling OLS: {e}")
                # Retourner des bêtas par défaut
                betas = pd.DataFrame(index=group.index, 
                                   columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
                betas['Mkt-RF'] = 1.0
                betas[['SMB', 'HML', 'RMW', 'CMA']] = 0.0
                return betas
        
        betas = factor_data_with_returns.groupby('ticker', group_keys=False).apply(safe_rolling_ols)
        return betas
        
    except Exception as e:
        print(f"Erreur calcul bêtas: {e}")
        # Retourner des bêtas par défaut pour toutes les actions
        tickers = data.index.get_level_values('ticker').unique()
        dates = data.index.get_level_values('date').unique()
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        betas = pd.DataFrame(index=index, columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        betas['Mkt-RF'] = 1.0
        betas[['SMB', 'HML', 'RMW', 'CMA']] = 0.0
        return betas

def perform_clustering(data):
    """Effectue le clustering avec gestion d'erreur robuste"""
    print("Clustering des données...")
    
    # Définir les centroïdes cibles pour RSI
    target_rsi_values = [30, 45, 55, 70]
    
    def safe_get_clusters(group_df):
        try:
            if len(group_df) < 4:
                group_df['cluster'] = np.random.randint(0, 4, len(group_df))
                return group_df
            
            # Préparer les données pour le clustering
            cluster_features = group_df.select_dtypes(include=[np.number]).fillna(group_df.select_dtypes(include=[np.number]).mean())
            
            if cluster_features.empty:
                group_df['cluster'] = 0
                return group_df
            
            # Vérifier si RSI existe
            if 'rsi' in cluster_features.columns:
                rsi_column_index = cluster_features.columns.get_loc('rsi')
                initial_centroids = np.zeros((4, len(cluster_features.columns)))
                initial_centroids[:, rsi_column_index] = target_rsi_values
                
                # Normaliser les autres colonnes des centroïdes
                for i, col in enumerate(cluster_features.columns):
                    if col != 'rsi':
                        col_mean = cluster_features[col].mean()
                        initial_centroids[:, i] = col_mean
            else:
                # Si pas de RSI, utiliser des centroïdes aléatoires
                initial_centroids = 'k-means++'
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=4, random_state=42, init=initial_centroids, n_init=1 if isinstance(initial_centroids, np.ndarray) else 10)
            group_df['cluster'] = kmeans.fit_predict(cluster_features)
            
            return group_df
            
        except Exception as e:
            print(f"Erreur clustering: {e}")
            # Assigner des clusters aléatoirement
            group_df['cluster'] = np.random.randint(0, 4, len(group_df))
            return group_df
    
    try:
        clustered_data = data.groupby('date', group_keys=False).apply(safe_get_clusters)
        return clustered_data
    except Exception as e:
        print(f"Erreur clustering global: {e}")
        data['cluster'] = 0  # Cluster par défaut
        return data

def create_portfolios(data):
    """Crée les portfolios basés sur le clustering"""
    print("Création des portfolios...")
    
    try:
        # Filtrer le cluster 3 (high RSI)
        filtered_df = data[data['cluster'] == 3].copy()
        
        if filtered_df.empty:
            print("Aucune action dans le cluster 3, utilisation du cluster 2")
            filtered_df = data[data['cluster'] == 2].copy()
        
        if filtered_df.empty:
            print("Utilisation de toutes les données")
            filtered_df = data.copy()
        
        # Décaler l'index d'un mois
        filtered_df = filtered_df.reset_index(level=1)
        filtered_df.index = filtered_df.index + pd.DateOffset(months=1)
        filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
        
        # Créer le dictionnaire des dates
        dates = filtered_df.index.get_level_values('date').unique().tolist()
        fixed_dates = {}
        
        for d in dates:
            tickers_for_date = filtered_df.xs(d, level=0).index.tolist()
            if len(tickers_for_date) >= 2:  # Au moins 2 actions pour un portfolio
                fixed_dates[d.strftime('%Y-%m-%d')] = tickers_for_date
        
        print(f"Nombre de dates de formation de portfolio: {len(fixed_dates)}")
        return fixed_dates, filtered_df
        
    except Exception as e:
        print(f"Erreur création portfolios: {e}")
        return {}, pd.DataFrame()

def optimize_portfolio_weights(prices, lower_bound=0):
    """Optimise les poids du portfolio avec gestion d'erreur"""
    try:
        if len(prices.columns) < 2:
            return {prices.columns[0]: 1.0}
        
        returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
        cov = risk_models.sample_cov(prices=prices, frequency=252)
        
        ef = EfficientFrontier(expected_returns=returns,
                             cov_matrix=cov,
                             weight_bounds=(lower_bound, 0.1),
                             solver='SCS')
        weights = ef.max_sharpe()
        return ef.clean_weights()
    
    except Exception as e:
        print(f"Optimisation failed: {e}")
        # Poids égaux comme fallback
        n_assets = len(prices.columns)
        return {col: 1/n_assets for col in prices.columns}

def calculate_portfolio_returns(fixed_dates, data):
    """Calcule les rendements du portfolio"""
    print("Calcul des rendements du portfolio...")
    
    try:
        # Télécharger les prix quotidiens
        stocks = data.index.get_level_values('ticker').unique().tolist()
        
        start_download = data.index.get_level_values('date').min() - pd.DateOffset(months=12)
        end_download = data.index.get_level_values('date').max()
        
        print(f"Téléchargement des prix quotidiens pour {len(stocks)} actions...")
        daily_prices = yf.download(tickers=stocks, start=start_download, end=end_download)
        
        if 'Adj Close' in daily_prices.columns:
            daily_prices = daily_prices['Adj Close']
        else:
            daily_prices = daily_prices.xs('Adj Close', axis=1, level=1)
        
        # Calculer les rendements quotidiens
        returns_dataframe = np.log(daily_prices).diff().dropna()
        
        portfolio_df = pd.DataFrame()
        
        for start_date in list(fixed_dates.keys())[:12]:  # Limiter à 12 mois pour les tests
            try:
                end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                cols = fixed_dates[start_date]
                
                optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
                optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                
                # Données pour l'optimisation
                optimization_df = daily_prices.loc[optimization_start_date:optimization_end_date, cols]
                optimization_df = optimization_df.dropna(axis=1)
                
                if len(optimization_df.columns) < 2:
                    continue
                
                # Optimiser les poids
                weights = optimize_portfolio_weights(optimization_df, 
                                                   lower_bound=round(1/(len(optimization_df.columns)*2), 3))
                
                # Calculer les rendements pondérés
                temp_returns = returns_dataframe.loc[start_date:end_date, list(weights.keys())]
                
                if temp_returns.empty:
                    continue
                
                # Calcul des rendements du portfolio
                portfolio_returns = (temp_returns * pd.Series(weights)).sum(axis=1)
                portfolio_returns = portfolio_returns.to_frame('Strategy Return')
                
                portfolio_df = pd.concat([portfolio_df, portfolio_returns], axis=0)
                
            except Exception as e:
                print(f"Erreur traitement {start_date}: {e}")
                continue
        
        return portfolio_df.drop_duplicates()
        
    except Exception as e:
        print(f"Erreur calcul rendements portfolio: {e}")
        return pd.DataFrame()

def main():
    """Fonction principale"""
    print("🚀 Démarrage de la stratégie de trading par clustering")
    print("=" * 60)
    
    try:
        # 1. Télécharger la liste S&P500
        print("\n1️⃣ Téléchargement de la liste S&P500...")
        symbols_list = safe_download_sp500()
        print(f"Actions sélectionnées: {len(symbols_list)}")
        
        # 2. Définir les dates
        end_date = '2023-09-27'
        start_date = pd.to_datetime(end_date) - pd.DateOffset(365*3)  # Réduire à 3 ans pour les tests
        
        # 3. Télécharger les données
        print(f"\n2️⃣ Téléchargement des données ({start_date.strftime('%Y-%m-%d')} à {end_date})...")
        df = download_stock_data(symbols_list, start_date, end_date)
        print(f"Données téléchargées: {df.shape}")
        
        # 4. Calculer les indicateurs techniques
        print("\n3️⃣ Calcul des indicateurs techniques...")
        df = calculate_technical_indicators(df)
        
        # 5. Agrégation mensuelle
        print("\n4️⃣ Agrégation mensuelle...")
        data = aggregate_monthly_data(df)
        print(f"Données mensuelles: {data.shape}")
        
        # 6. Calcul des rendements
        print("\n5️⃣ Calcul des rendements...")
        data = calculate_returns(data)
        print(f"Données avec rendements: {data.shape}")
        
        # 7. Facteurs Fama-French
        print("\n6️⃣ Téléchargement facteurs Fama-French...")
        factor_data = get_fama_french_factors()
        
        # 8. Calcul des bêtas
        print("\n7️⃣ Calcul des bêtas factoriels...")
        betas = calculate_factor_betas(data, factor_data)
        
        # 9. Joindre les bêtas aux données
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        data = data.join(betas.groupby('ticker').shift())
        data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
        
        # Nettoyer les données
        if 'close' in data.columns:
            data = data.drop('close', axis=1)
        data = data.dropna()
        print(f"Données finales: {data.shape}")
        
        # 10. Clustering
        print("\n8️⃣ Clustering des données...")
        data = perform_clustering(data)
        print("Distribution des clusters:")
        print(data['cluster'].value_counts().sort_index())
        
        # 11. Création des portfolios
        print("\n9️⃣ Création des portfolios...")
        fixed_dates, filtered_df = create_portfolios(data)
        
        if not fixed_dates:
            print("❌ Aucun portfolio créé")
            return
        
        # 12. Calcul des rendements du portfolio
        print("\n🔟 Calcul des rendements du portfolio...")
        portfolio_df = calculate_portfolio_returns(fixed_dates, data)
        
        if portfolio_df.empty:
            print("❌ Aucun rendement de portfolio calculé")
            return
        
        # 13. Benchmark (SPY)
        print("\n1️⃣1️⃣ Téléchargement du benchmark...")
        spy = yf.download('SPY', start=start_date, end=end_date)
        spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close': 'SPY Buy&Hold'}, axis=1)
        
        # 14. Fusionner les rendements
        portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='inner')
        
        # 15. Affichage des résultats
        print("\n1️⃣2️⃣ Résultats de la stratégie...")
        if len(portfolio_df) > 0:
            display_results(portfolio_df)
        else:
            print("❌ Aucune donnée de portfolio à afficher")
            
        return portfolio_df
        
    except Exception as e:
        print(f"❌ Erreur dans la fonction principale: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def display_results(portfolio_df):
    """Affiche les résultats de la stratégie"""
    try:
        plt.style.use('ggplot')
        
        # Calcul des rendements cumulés
        portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1
        
        # Graphique
        plt.figure(figsize=(16, 8))
        portfolio_cumulative_return.plot()
        plt.title('Stratégie de Trading par Clustering - Rendements Cumulés', fontsize=16)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.ylabel('Rendement Cumulé')
        plt.xlabel('Date')
        plt.legend(['Stratégie', 'S&P 500 (SPY)'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiques de performance
        strategy_returns = portfolio_df['Strategy Return']
        spy_returns = portfolio_df['SPY Buy&Hold']
        
        strategy_total_return = portfolio_cumulative_return['Strategy Return'].iloc[-1]
        spy_total_return = portfolio_cumulative_return['SPY Buy&Hold'].iloc[-1]
        
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        spy_volatility = spy_returns.std() * np.sqrt(252)
        
        strategy_sharpe = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        spy_sharpe = (spy_returns.mean() * 252) / (spy_returns.std() * np.sqrt(252))
        
        # Calcul du maximum drawdown
        def calculate_max_drawdown(returns):
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            return drawdown.min()
        
        strategy_max_dd = calculate_max_drawdown(strategy_returns)
        spy_max_dd = calculate_max_drawdown(spy_returns)
        
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DE PERFORMANCE DE LA STRATÉGIE")
        print("="*80)
        print(f"📈 Rendement Total Stratégie:     {strategy_total_return:>10.2%}")
        print(f"📈 Rendement Total S&P 500:      {spy_total_return:>10.2%}")
        print(f"🎯 Alpha (Surperformance):       {strategy_total_return - spy_total_return:>10.2%}")
        print("-" * 80)
        print(f"📊 Volatilité Stratégie:         {strategy_volatility:>10.2%}")
        print(f"📊 Volatilité S&P 500:           {spy_volatility:>10.2%}")
        print("-" * 80)
        print(f"⚡ Ratio de Sharpe Stratégie:    {strategy_sharpe:>10.2f}")
        print(f"⚡ Ratio de Sharpe S&P 500:      {spy_sharpe:>10.2f}")
        print("-" * 80)
        print(f"📉 Max Drawdown Stratégie:       {strategy_max_dd:>10.2%}")
        print(f"📉 Max Drawdown S&P 500:         {spy_max_dd:>10.2%}")
        print("-" * 80)
        print(f"📅 Période d'analyse:            {portfolio_df.index[0].strftime('%Y-%m-%d')} à {portfolio_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"🔢 Nombre d'observations:        {len(portfolio_df):>10}")
        
        # Statistiques mensuelles
        monthly_wins_strategy = (strategy_returns > 0).sum()
        monthly_wins_spy = (spy_returns > 0).sum()
        total_months = len(strategy_returns)
        
        print("-" * 80)
        print(f"🎯 Mois gagnants Stratégie:      {monthly_wins_strategy}/{total_months} ({monthly_wins_strategy/total_months:.1%})")
        print(f"🎯 Mois gagnants S&P 500:        {monthly_wins_spy}/{total_months} ({monthly_wins_spy/total_months:.1%})")
        print("="*80)
        
        # Analyse des corrélations
        correlation = strategy_returns.corr(spy_returns)
        print(f"🔗 Corrélation avec S&P 500:     {correlation:>10.2f}")
        
        # Beta de la stratégie
        beta = np.cov(strategy_returns, spy_returns)[0, 1] / np.var(spy_returns)
        alpha_annual = (strategy_returns.mean() - beta * spy_returns.mean()) * 252
        
        print(f"📈 Bêta de la stratégie:         {beta:>10.2f}")
        print(f"🎯 Alpha annualisé:              {alpha_annual:>10.2%}")
        print("="*80)
        
        # Analyse par année si assez de données
        if len(portfolio_df) > 12:
            print("\n📅 PERFORMANCE ANNUELLE:")
            print("-" * 50)
            yearly_returns = portfolio_cumulative_return.resample('Y').last().pct_change().dropna()
            for year, row in yearly_returns.iterrows():
                year_str = year.strftime('%Y')
                print(f"{year_str}  |  Stratégie: {row['Strategy Return']:>8.2%}  |  S&P 500: {row['SPY Buy&Hold']:>8.2%}")
        
        return {
            'total_return_strategy': strategy_total_return,
            'total_return_spy': spy_total_return,
            'alpha': strategy_total_return - spy_total_return,
            'volatility_strategy': strategy_volatility,
            'sharpe_strategy': strategy_sharpe,
            'max_drawdown_strategy': strategy_max_dd,
            'correlation': correlation,
            'beta': beta
        }
        
    except Exception as e:
        print(f"❌ Erreur dans l'affichage des résultats: {e}")
        print("Données du portfolio disponibles:")
        print(portfolio_df.head())

def plot_cluster_analysis(data):
    """Analyse et visualisation des clusters"""
    try:
        plt.style.use('ggplot')
        
        # Vérifier les colonnes nécessaires
        if 'garman_klass_vol' not in data.columns or 'rsi' not in data.columns or 'cluster' not in data.columns:
            print("❌ Colonnes manquantes pour l'analyse des clusters")
            return
        
        # Prendre un échantillon de dates pour l'analyse
        sample_dates = data.index.get_level_values('date').unique()[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['red', 'green', 'blue', 'orange']
        cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3 (High RSI)']
        
        for i, date in enumerate(sample_dates):
            if i >= 6:
                break
                
            try:
                date_data = data.xs(date, level=0)
                
                ax = axes[i]
                
                for cluster in range(4):
                    cluster_data = date_data[date_data['cluster'] == cluster]
                    if len(cluster_data) > 0:
                        ax.scatter(cluster_data['garman_klass_vol'], 
                                 cluster_data['rsi'],
                                 c=colors[cluster], 
                                 label=cluster_names[cluster], 
                                 alpha=0.7,
                                 s=50)
                
                ax.set_xlabel('Volatilité Garman-Klass')
                ax.set_ylabel('RSI')
                ax.set_title(f'Clusters - {date.strftime("%Y-%m")}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Erreur plot pour {date}: {e}")
                continue
        
        plt.tight_layout()
        plt.suptitle('Analyse des Clusters par Mois', fontsize=16, y=1.02)
        plt.show()
        
        # Statistiques des clusters
        print("\n📊 STATISTIQUES DES CLUSTERS:")
        print("=" * 60)
        
        cluster_stats = data.groupby('cluster').agg({
            'rsi': ['mean', 'std', 'count'],
            'garman_klass_vol': ['mean', 'std'],
            'return_1m': ['mean', 'std']
        }).round(3)
        
        print(cluster_stats)
        
        # Distribution des clusters dans le temps
        plt.figure(figsize=(14, 6))
        cluster_counts = data.groupby(['date', 'cluster']).size().unstack(fill_value=0)
        cluster_counts.plot(kind='area', stacked=True, 
                           colors=colors, alpha=0.7)
        plt.title('Évolution de la Distribution des Clusters dans le Temps')
        plt.xlabel('Date')
        plt.ylabel('Nombre d\'Actions par Cluster')
        plt.legend(cluster_names)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur dans l'analyse des clusters: {e}")

# Point d'entrée du programme
if __name__ == "__main__":
    # Exécuter la stratégie
    portfolio_results = main()
    
    # Si des résultats sont disponibles, faire l'analyse des clusters

    print("\n🔚 FIN DU PROGRAMME")