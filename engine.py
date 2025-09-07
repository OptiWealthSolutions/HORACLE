from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as pltxs
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
import seaborn as sns

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

TickerList = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCAD=X","AUDUSD=X","NZDUSD=X"]

# --- Data Loading, Cleaning and processing ---
def getDataLoad(ticker,period,interval):
    df = yf.download(ticker, period=period, progress=False,interval=interval)
    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")
    df = df.copy()
    df = df.dropna()
    return df

# --- Features Creating, Engineering, Comparaison and Selection ---
# Linearity, Normality, 
# Homoscedasticity,
# Stationarity, 
def getDataStationarityTest(df, feature_name: str, significance=0.05):
    series = df[feature_name].dropna()
    result = adfuller(series)
    p_value = result[1]
    stationary = p_value < significance
    return {"p_value": p_value, "stationary": stationary}

# function which compute features 
def computeFeatures(df):
    #standardisation
    def getDataStandardized (df):
        df = df.copy()
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df #ne pas strandardiser les features deja en % ou deja normalisées

    def getSMA(df: pd.DataFrame, period):
            # moving average features
            df[f'Mov_av_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'Mov_av_{period}'] = getDataStandardized(df[f'Mov_av_{period}'])
            return df

    def getRSI(df: pd.DataFrame, period):
            delta = df['Close'].diff()  
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'] = rsi
            return df

    def getMACD(df):
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['MACD'] = macd_line
            df['MACD_signal'] = signal_line
            getDataStandardized(df[f'MACD_signal'])
            return df

    def getSTOCH(df):
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            stoch = 100 * (df['Close'] - low14) / (high14 - low14)
            df['STOCH'] = stoch
            getDataStandardized(df[f'STOCH'])
            return df

    def getMomentumFactors(df,n):
        for lag in range(1, n):
            df[f'momentum_{lag}'] = df['Close'].pct_change(periods=lag)
        return df

    def getMomentumPeriod(df,n):
        prices = df['Close']
        monthly_prices = prices.resample('M').last()
        for lag in range(1, n):
            df[f'return_{lag}m'] = (monthly_prices
            .pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
            .add(1)
            .pow(1/lag)
            .sub(1))
            getDataStandardized(df[f'return_{lag}m'])
        return df

    def getVolumeStandardize(df):
        df['Volume'] = getDataStandardized(df['Volume'])
        return df

    def getFeaturesClean(df):
        df = df.dropna()
        df.drop('High', axis=1, inplace=True)
        df.drop('Low', axis=1, inplace=True)
        df.drop('Open', axis=1, inplace=True)
        return df

    df = getSMA(df, 14)
    df = getRSI(df, 14)
    df = getMACD(df)
    df = getSTOCH(df)
    df = getMomentumFactors(df, 5)
    df = getMomentumPeriod(df, 3)
    df = getVolumeStandardize(df)
    df = getFeaturesClean(df)
    return df
# Multicollinearity 
#covariance matrix for testing colinearity
def getCovMatrix(df,featureslist):
    df = df[featureslist].dropna()
    cov_matrix_df = df.cov()
    sns.heatmap(cov_matrix_df, annot=True, cmap='coolwarm')
    plt.show()
    return 
# --- Labelling Engineering ---
# Mutliclass Threshold Labelling and Triple barrier labelling
def getDailyVol(df,span=100):
    df = df['Close'].index.searchsorted(df['Close'].index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = (pd.Series(df.index[df - 1], 
                   index=df.index[df.shape[0] - df.shape[0]:]))
    
    df = df.loc[df.index] / df.loc[df.values].values - 1  # daily rets
    df = df.ewm(span=span).std()
    return df

def getEvents(df, threshold):
    close = df['Close']
    t_events, s_pos, s_neg = [], 0, 0
    diff = np.log(close).diff().dropna()
    
    for i in diff.index[1:]:
        pos, neg = float(s_pos + diff.loc[i]), float(s_neg + diff.loc[i])
        s_pos, s_neg = max(0.0, pos), min(0.0, neg)
        
        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)
    
    return pd.DatetimeIndex(t_events)

def getSingleBarrier(close, loc, t1, pt, sl):
    t0 = loc  # Event start time
    prices = close[loc:t1]  # Prix sur la période
    
    if len(prices) < 2:
        return pd.NaT, np.nan
    
    cum_rets = (prices / close[t0] - 1.0)
    
    for timestamp, ret in cum_rets.items():
        if timestamp == t0:
            continue
            
        if pd.notna(pt[t0]) and ret >= pt[t0]:
            return timestamp, 1  # Profit taking
        
        if pd.notna(sl[t0]) and ret <= sl[t0]:
            return timestamp, -1  # Stop loss
    
    return t1, 0  # Time barrier


def getTripleBarrierLabels(events, min_ret):
    #min_return_ajd adapted to the volatility
    close = events['Close']
    events['return_std'] = close.pct_change().std()
    events['return_mean'] = close.pct_change().mean()
    min_return_adj = events['return_mean'] * (2*events['return_std'])
    bins = events['ret'].copy()
    bins[bins >= min_return_adj] = 1  # Profit
    bins[bins <= -min_return_adj] = -1  # Loss
    bins[(bins < min_return_adj) & (bins > -min_return_adj)] = 0  # Neutral
    
    return bins

def applyTripleBarrier(df, pt=0.02, sl=0.02, min_ret=0.01):
    close = df['Close']
    threshold = 0.01
    t_events = getEvents(df, threshold)
    pt_levels = pd.Series(pt, index=t_events)
    sl_levels = pd.Series(-sl, index=t_events)
    events = pd.DataFrame(index=t_events)
    events['pt_level'] = pt_levels
    events['sl_level'] = sl_levels
    events['ret'] = 0.0
    for loc in t_events:
        t1 = loc + pd.Timedelta(days=5)
        if t1 > close.index[-1]:
            t1 = close.index[-1]
        end_time, label = getSingleBarrier(close, loc, t1, pt_levels, sl_levels)
        events.at[loc, 'ret'] = close[end_time] / close[loc] - 1
    bins = getTripleBarrierLabels(events, min_ret)
    return bins

# --- Meta-Labelling ---

def getMetaFeatures(primary_model, X, y, close, events):
    # primary model pred
    primary_predictions = primary_model.predict(X)
    primary_probabilities = primary_model.predict_proba(X)
    
    # entropy computing (confidence of the signal)
    def computeEntropy(probabilities):
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
        return entropy / np.log2(probabilities.shape[1])  # Normaliser
    
    confidence = 1 - computeEntropy(primary_probabilities)
    
    # consistency
    def calculateConsistency(predictions, window=5):
        consistency = []
        for i in range(len(predictions)):
            start_idx = max(0, i - window)
            end_idx = min(len(predictions), i + window + 1)
            local_preds = predictions[start_idx:end_idx]
            consistency.append(np.std(local_preds))
        return np.array(consistency)
    
    consistency = calculateConsistency(primary_predictions)
    
    vol_features = []
    momentum_features = []
    
    for event_time in events.index:
        # Volatilité récente
        vol_window = close.loc[:event_time].tail(20).pct_change().std()
        vol_features.append(vol_window)
        
        # Momentum récent
        momentum = (close.loc[event_time] / close.loc[:event_time].tail(5).iloc[0]) - 1
        momentum_features.append(momentum)
    
    # barrier_distances
    barrier_distances = []
    for idx, event in events.iterrows():
        current_price = close.loc[idx]
        if 'pt_level' in event and pd.notna(event['pt_level']):
            pt_distance = abs(event['pt_level'] - current_price) / current_price
        else:
            pt_distance = 0.05  # Default
            
        if 'sl_level' in event and pd.notna(event['sl_level']):
            sl_distance = abs(current_price - event['sl_level']) / current_price
        else:
            sl_distance = 0.05  # Default
            
        barrier_distances.append(min(pt_distance, sl_distance))
    
    # meta features
    meta_features = pd.DataFrame({
        'primary_pred': primary_predictions,
        'confidence': confidence,
        'consistency': consistency,
        'volatility': vol_features,
        'momentum': momentum_features,
        'barrier_distance': barrier_distances,
        'max_prob': np.max(primary_probabilities, axis=1),
        'prob_spread': np.max(primary_probabilities, axis=1) - np.median(primary_probabilities, axis=1)
    }, index=events.index)
    
    return meta_features
# --- Model(s) Building ---
#Primary Model : "Random Forest Classifier"
def RandomForestClassifier_(features_df,target):
    #features and target
    X = features_df
    y = target

    #train test split
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    
    #model RFC
    PrimaryModel = RandomForestClassifier(
        n_estimators=300,      # 300 arbres
        max_depth=10,          # Profondeur max = 10
        min_samples_split=50,  # Min 50 échantillons pour split
        min_samples_leaf=20,   # Min 20 échantillons par feuille
        max_features='sqrt',   # √(nb_features) features par split
        #bootstrap=True,        # Bootstrap sampling
        #oob_score=True,        # Out-of-bag score
        class_weight='balanced', # Équilibrer les classes
        random_state=42
    )

    #cross_validation
    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_score(
        PrimaryModel,
        X_train,
        y_train,
        cv=tscv,
        scoring='accuracy'
    )

    #fit model
    PrimaryModel.fit(X_train,y_train)
    PrimaryModel.score(X_test,y_test)
    
    #metrics
    confusion_matrix_ = confusion_matrix(y_test,PrimaryModel.predict(X_test))
    print(f"Confusion matrix: {confusion_matrix_}")
    classification_report_ = classification_report(y_test,PrimaryModel.predict(X_test))
    print(f"Classification report: {classification_report_}")
    

    return PrimaryModel.predict(X_test)

#Meta-labelling Model
def MetaModel(meta_features, events, threshold_profit=0.02):
    # Création des labels meta (binaire: trade profitable ou non)
    MetaLabels = (events['ret'].abs() >= threshold_profit).astype(int)
    
    #train test split
    X_train,X_test,y_train,y_test = train_test_split(
        meta_features,MetaLabels,test_size=0.2,random_state=42
    )
    
    MetaModel_ = RandomForestClassifier(
        n_estimators=100,     # Plus simple que le modèle principal
        max_depth=5,          # Moins profond
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
        #cross_validation
    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_score(
        MetaModel_,
        X_train,
        y_train,
        cv=tscv,
        scoring='accuracy'
    )

    #fit model
    MetaModel_.fit(X_train,y_train)
    MetaModel_.score(X_test,y_test)
    
    confusion_matrix_meta = confusion_matrix(y_test,MetaModel_.predict(X_test))
    print(f"Confusion matrix: {confusion_matrix_meta}")
    classification_report_meta = classification_report(y_test,MetaModel_.predict(X_test))
    print(f"Classification report: {classification_report_meta}")
    
    return MetaModel_.predict(X_test)

# --- Model(s) Training, Testing, Validation ---
# train-test split

#cross-validation

#hyper params

# --- Model(s) measurement --- 
#classification report
#confusion matrix

# --- BackTesting ---


# --- Main function activation ---

def main(ticker):
    df = getDataLoad(ticker, period="2y", interval="1d")
    df = computeFeatures(df)
    df = df.dropna()
    bins = applyTripleBarrier(df)
    bins = bins.loc[df.index.intersection(bins.index)]
    features = df.loc[bins.index]
    RandomForestClassifier_(features, bins)

if __name__ == "__main__":
    for el in TickerList:
        main(el)
    
    