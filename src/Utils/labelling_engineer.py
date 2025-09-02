import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))

class LabellingEngineer:
    def __init__(self):
        self.df = None
        pass

    def labelling_simple (self, df: pd.DataFrame,Shift : int) -> pd.DataFrame:
        self.df = df
        self.df['Target'] = self.df['Close'].shift(-Shift)
        return self.df

    def meta_labelling(self):
        
        return
    
    def triple_barrier_method(self):
        return

    def trend_following_labelling(self):
        return
        
    def adaptative_threshold(self):

        return

    def fixed_threshold(self):
        return
    
    def mutliclass_classification(self):
        return

class Labbelling_engineer_model_fitting():
    def __init__(self):
        self.df = None

# ----- classifier labelling method ---- (standar_scale all the features before labelling method)
    def classifier__binary_labelling_method(df):
        df['TARGET'] = df['Close'].astype(int)
        return 

    def classifier_threshold_labelling_method(df, seuil_lambda):
        #le seuil sera à definir dans les paramètres du modele
        df['TARGET'] = (df['Close']>seuil_lambda).astype(int)
        return 

    def classifier_multiclass_percentil_labelling_method(df,percentile_n):
        df['TARGET'] = np.qcut(df['Close'], q=percentile_n,labels=[0,1,2])

        return

# ----- random forest labelling method ----
#we have to give an input threshold to the random forest, this will be a list type 
    def rand_forest_multiclass_labelling_method(df,threshold = List[float]):
        returns = df['Close'].pct_change()
        df['TARGET'] = np.where(returns < threshold[0], 1, np.where(returns > threshold[1], 2, 1))
        return
        
    def rand_forest_labelling_threshold(df,prediction_horizon):
        df = df.copy()
        
        # Calcul du rendement futur sur prediction_horizon jours
        future_returns = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        
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
        
        print(f"Distribution des labels:")
        print(pd.Series(labels).value_counts().sort_index())
        
        return df


    def rand_forest_quantil_labelling_method(df, quantil = List[float]):
        returns = df['Close'].pct_change()

        df['TARGET'] = np.qcut(returns, q=quantil, labels=range(5)) #0,1,2,3,4 = labels'names
        return


# ----- classifier labelling method ----
    def logistic_reg_labelling_method(df):
        return 
        
    def lin_reg_labelling_method(df):
        return