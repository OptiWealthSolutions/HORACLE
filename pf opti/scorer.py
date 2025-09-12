import yfinance as yf
import numpy as np
import pandas as pd

#parameters and test variables
teststocksList = ["AAPL", "MSFT", "AMZN", "TSLA", "META"]
class DataLoader():
    def __init__(self):
        self.tickers_list = ["AAPL", "MSFT", "AMZN", "TSLA", "META"]
        self.PERIOD = "20y"
        self.INTERVAL = "1d"

    def getDataLoad(self):
        self.df = pd.DataFrame()
        self.df = yf.download(self.tickers_list, period=self.PERIOD, interval=self.INTERVAL)['Close']
        self.df = self.df.dropna()
        self.df['return'] = self.df['return'].pct_change().dropna()
        self.df = self.df['return'].resample('Q').stack()
        self.df['annual_return'] = self.df['return'].rolling(252).mean().dropna()
        return self.df

    def getMacroData(self):
        #fred api (this function is optionnal because of issue with fredapi)
        return
    
    def getFundamentalData(self):
        #PE ratio, ROE, debt to equity, revenue growth, roic, current ratio
        for ticker in self.tickers_list:
            self.df['PE_ratio'] = yf.Ticker(ticker).info['trailingPE']
            self.df['ROE'] = yf.Ticker(ticker).info['returnOnEquity']
            self.df['debt_to_equity'] = yf.Ticker(ticker).info['debtToEquity']
            self.df['revenue_growth'] = yf.Ticker(ticker).info['revenueGrowth']
            self.df['roic'] = yf.Ticker(ticker).info['returnOnInvestedCapital']
            print(self.df)
        return

    def getCompaniesFiltered(self):
        '''
        - Return on equity ≥ 20
        - ROIC ≥ 15%
        - Debt to equity ratio ≥ 1.5
        - Current ratio ≥ 1.5
        - Market cap ≥ 1BN
        - PE ratio ≤ average P/E ratio sur 10Y
        - the last year return has to be positive or near to 0 if it is not
        '''
        self.df = self.df[(self.df['trainlingPE'] <= self.df['trainlingPE'].mean()) & 
        (self.df['ROE'] >= 20) & (self.df['ROIC'] >= 15) & (self.df['debt_to_equity'] >= 1.5) & 
        (self.df['current_ratio'] >= 1.5)]
        self.df = self.df[(self.df['annual_return'] >= 0) | (self.df['annual_return'] <= 0.01)]
        print(self.df)
        return 

def main():
    dataLoader = DataLoader()
    dataLoader.getDataLoad()
    dataLoader.getCompaniesFiltered()
    return

def __init__():
    main()
