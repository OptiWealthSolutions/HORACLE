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
        # Download adjusted close prices
        self.df = yf.download(self.tickers_list, period=self.PERIOD, interval=self.INTERVAL)["Close"]
        self.df = self.df.dropna()
        # Compute daily returns and rolling annualized return
        returns = self.df.pct_change().dropna()
        self.df["return"] = returns.mean(axis=1)  # moyenne des rendements
        self.df["annual_return"] = self.df["return"].rolling(252).mean()
        return self.df

    def getMacroData(self):
        #fred api (this function is optionnal because of issue with fredapi)
        return
    
    def getFundamentalData(self):
        # Collect fundamentals in a new DataFrame
        fundamentals = []
        for ticker in self.tickers_list:
            info = yf.Ticker(ticker).info
            fundamentals.append({
                "Ticker": ticker,
                "PE_ratio": info.get("trailingPE", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "debt_to_equity": info.get("debtToEquity", np.nan),
                "revenue_growth": info.get("revenueGrowth", np.nan),
                "ROIC": info.get("returnOnInvestedCapital", np.nan),
                "current_ratio": info.get("currentRatio", np.nan),
                "market_cap": info.get("marketCap", np.nan)
            })
        self.fundamentals = pd.DataFrame(fundamentals)
        return self.fundamentals

    def getCompaniesFiltered(self):
        '''
        - Return on equity ≥ 20%
        - ROIC ≥ 15%
        - Debt to equity ratio ≤ 1.5
        - Current ratio ≥ 1.5
        - Market cap ≥ 1BN
        - PE ratio ≤ average P/E ratio
        '''
        if self.fundamentals is None:
            raise ValueError("Fundamentals not loaded. Run getFundamentalData() first.")

        avg_pe = self.fundamentals["PE_ratio"].rolling(2552).mean()
        filtered = self.fundamentals[
            (self.fundamentals["PE_ratio"] <= avg_pe) &
            (self.fundamentals["ROE"] >= 0.20) &
            (self.fundamentals["ROIC"] >= 0.15) &
            (self.fundamentals["debt_to_equity"] <= 1.5) &
            (self.fundamentals["current_ratio"] >= 1.5) &
            (self.fundamentals["market_cap"] >= 1e9)
        ]
        return filtered

def main():
    dataLoader = DataLoader()
    prices = dataLoader.getDataLoad()
    fundamentals = dataLoader.getFundamentalData()
    filtered = dataLoader.getCompaniesFiltered()

    print("Prices head:")
    print(prices.head())
    print("\nFundamentals:")
    print(fundamentals)
    print("\nFiltered Companies:")
    print(filtered)
    return

if __name__ == "__main__":
    main()
