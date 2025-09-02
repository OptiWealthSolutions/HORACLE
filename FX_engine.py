import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Utils.data_engineer import *
from src.Utils.feature_engineer import Tech_FeatureEngineer, Macro_FeatureEngineer, Quant_FeatureEngineer
from src.Utils.labelling_engineer import LabellingEngineer

# Suppress warnings
warnings.filterwarnings('ignore')
load_dotenv()

class FXEngine:
    def __init__(self, ticker='EURUSD=X', period='15y', interval='1d'):
        """
        Initialize the FX Engine
        
        Args:
            ticker (str): Ticker symbol for the currency pair (e.g., 'EURUSD=X')
            period (str): Time period for historical data (e.g., '15y' for 15 years)
            interval (str): Data interval ('1d' for daily, '1h' for hourly, etc.)
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load and preprocess the FX data
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame with OHLCV data
        """
        print(f"Loading data for {self.ticker}...")
        try:
            # Download data using yfinance
            df = yf.download(
                self.ticker, 
                period=self.period, 
                interval=self.interval,
                progress=False
            )
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
                
            # Basic cleaning
            df = self._clean_data(df)
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _clean_data(self, df):
        """
        Clean and preprocess the raw price data
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned and processed DataFrame
        """
        # Forward fill missing values
        df = df.ffill()
        
        # Remove outliers using IQR method
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)  # Annualized
        
        return df