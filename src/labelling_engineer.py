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