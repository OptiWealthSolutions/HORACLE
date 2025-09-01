import pandas as pd
import numpy as np

class LabellingEngineer:
    def __init__(self):
        self.df = None
        pass
    def labelling_simple (self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df
        self.df['Target'] = self.df['Close'].shift(-SHIFT)
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
    
    def rand_forest_quantil_labelling_method(df, quantil = List[float]):
        returns = df['Close'].pct_change()

        df['TARGET'] = np.qcut(future_return, q=5, labels=range(5)) #0,1,2,3,4 = labels'names
        return


# ----- classifier labelling method ----
    def logistic_reg_labelling_method(df):
        return 
        
    def lin_reg_labelling_method(df):
        return