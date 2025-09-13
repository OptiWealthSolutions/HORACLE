import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def getIVP(cov, **kwargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

def getQuasiDiag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i)//2), 
                 (len(i)//2, len(i))) if len(i) > 1]  # bi-section
        
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= (1 - alpha)  # weight 2
    return w

def correlDist(corr):
    dist = ((1 - corr) / 2.) ** 0.5  # distance matrix
    return dist

def plotCorrMatrix(corr, title="Correlation Matrix", labels=None):
    plt.figure(figsize=(10, 8))
    if labels is None:
        labels = corr.columns
    
    sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, 
                xticklabels=labels, yticklabels=labels, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return

def loadMarketData(tickers, period="20y", interval="1d"):
    print(f"Loading data for {len(tickers)} tickers...")
    data = yf.download(tickers, period=period, interval=interval, progress=False)
    if len(tickers) == 1:
        prices = data['Close'].to_frame()
        prices.columns = tickers
    else:
        prices = data['Close']
    # Clean data
    prices = prices.dropna()
    # Calculate returns
    returns = prices.pct_change().dropna()
    print(f"Data period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(returns)}")
    return prices, returns

class HRPOptimizer:
    def __init__(self, tickers=None, period="1y", interval="1d"):
        self.tickers = tickers
        
        # Load market data
        self.prices, self.returns = loadMarketData(self.tickers, period, interval)
        
        # Calculate covariance and correlation matrices
        self.cov = self.returns.cov() * 252  # Annualized
        self.corr = self.returns.corr()
        self.max_weight = 0.25
        # Store results
        self.hrp_weights = None
        self.sortIx = None
        
    def optimize(self):
        print("\n" + "="*50)
        print("HIERARCHICAL RISK PARITY OPTIMIZATION")
        print("="*50)
        
        # 1. Compute distance matrix
        dist = correlDist(self.corr)
        
        # 2. Perform hierarchical clustering
        link = sch.linkage(dist, 'single')
        
        # 3. Get quasi-diagonalization
        sortIx = getQuasiDiag(link)
        self.sortIx = self.corr.index[sortIx].tolist()  # recover labels
        
        # 4. Compute HRP weights
        self.hrp_weights = getRecBipart(self.cov, self.sortIx)
        
        print("HRP Optimization completed!")
        return self.hrp_weights
    
    def getIVPWeights(self):
        ivp_weights = getIVP(self.cov)
        return pd.Series(ivp_weights, index=self.cov.index)
    
    def getEqualWeights(self):
        n_assets = len(self.tickers)
        equal_weights = pd.Series([1/n_assets] * n_assets, index=self.tickers)
        return equal_weights
    
    def portfolioStats(self, weights):
        # Annualized return (using historical mean)
        mean_returns = self.returns.mean() * 252
        portfolio_return = np.sum(weights * mean_returns)
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))
        # Sharpe ratio (assuming risk-free rate = 3.464%)
        sharpe_ratio = (portfolio_return - 0.03464) / portfolio_vol if portfolio_vol != 0 else 0
        return {
            'Annual Return': portfolio_return,
            'Annual Volatility': portfolio_vol,
            'Sharpe Ratio': sharpe_ratio
        }
    
    def displayResults(self):
        if self.hrp_weights is None:
            print("Please run optimize() first!")
            return
        
        print("\n PORTFOLIO WEIGHTS COMPARISON")
        print("-" * 60)
        
        # Get comparison portfolios
        ivp_weights = self.getIVPWeights()
        equal_weights = self.getEqualWeights()
        
        # Create comparison DataFrame
        weights_df = pd.DataFrame({
            'HRP': self.hrp_weights * 100,
            'IVP': ivp_weights * 100,
            'Equal Weight': equal_weights * 100
        }).round(2)
        
        print(weights_df)
        
        # Calculate and display statistics
        print("\n PORTFOLIO STATISTICS")
        print("-" * 60)
        
        hrp_stats = self.portfolioStats(self.hrp_weights)
        ivp_stats = self.portfolioStats(ivp_weights)
        equal_stats = self.portfolioStats(equal_weights)
        
        stats_df = pd.DataFrame({
            'HRP': hrp_stats,
            'IVP': ivp_stats,
            'Equal Weight': equal_stats
        })
        
        # Format as percentages where appropriate
        stats_display = stats_df.copy()
        stats_display.loc['Annual Return'] *= 100
        stats_display.loc['Annual Volatility'] *= 100
        
        print(stats_display.round(3))
        
        # Risk contribution analysis for HRP
        print("\n HRP risk contribution analysis")
        print("-" * 60)
        self.analyzeRiskContribution()
        
    def analyzeRiskContribution(self):
        if self.hrp_weights is None:
            return
            
        portfolio_vol = np.sqrt(np.dot(self.hrp_weights.T, np.dot(self.cov, self.hrp_weights)))
        marginal_contrib = np.dot(self.cov, self.hrp_weights) / portfolio_vol
        risk_contrib = self.hrp_weights * marginal_contrib
        
        # Risk contribution as percentage
        risk_contrib_pct = (risk_contrib / risk_contrib.sum()) * 100
        
        risk_df = pd.DataFrame({
            'Weight (%)': self.hrp_weights * 100,
            'Risk Contribution (%)': risk_contrib_pct
        }).round(2)
        
        print(risk_df.sort_values('Risk Contribution (%)', ascending=False))
    
    def plotResults(self):
        if self.hrp_weights is None:
            print("Please run optimize() first!")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hierarchical Risk Parity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original correlation matrix
        sns.heatmap(self.corr, annot=True, cmap='RdYlBu_r', center=0,
                   ax=axes[0,0], fmt='.2f')
        axes[0,0].set_title('Original Correlation Matrix')
        
        # 2. Reordered correlation matrix
        if self.sortIx:
            reordered_corr = self.corr.loc[self.sortIx, self.sortIx]
            sns.heatmap(reordered_corr, annot=True, cmap='RdYlBu_r', center=0,
                       ax=axes[0,1], fmt='.2f')
            axes[0,1].set_title('Clustered Correlation Matrix')
        
        # 3. Dendrogram
        dist = correlDist(self.corr)
        link = sch.linkage(dist, 'single')
        '''
        A dendrogram (or clustering dendrogram) is a diagram that shows the hierarchical 
        relationship between objects. It is most commonly created as an output from hierarchical clustering. 
        Dendrograms are used in machine learning and data science to help visualize clustering.
        '''
        dendro = sch.dendrogram(link, labels=self.corr.index.tolist(), 
                               ax=axes[1,0], orientation='top')
        axes[1,0].set_title('Hierarchical Clustering Dendrogram')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Portfolio weights comparison
        ivp_weights = self.getIVPWeights()
        equal_weights = self.getEqualWeights()
        
        x = np.arange(len(self.tickers))
        width = 0.25
        
        axes[1,1].bar(x - width, self.hrp_weights * 100, width, label='HRP', alpha=0.8)
        axes[1,1].bar(x, ivp_weights * 100, width, label='IVP', alpha=0.8)
        axes[1,1].bar(x + width, equal_weights * 100, width, label='Equal Weight', alpha=0.8)
        
        axes[1,1].set_xlabel('Assets')
        axes[1,1].set_ylabel('Weight (%)')
        axes[1,1].set_title('Portfolio Weights Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(self.tickers)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    tickers = [
    "VOLV-B.ST",   # Volvo
    "CS.PA",       # AXA
    "RHM.DE",      # Rheinmetall
    "AM.PA",       # Dassault Aviation (ou Airbus si c'est AM : vérifier ton intention)
    "HO.PA",       # Thales (HO sur Euronext Paris)
    "CAP.PA",      # Capgemini
    "AAPL",        # Apple
    "CAT",         # Caterpillar
    "MA",          # Mastercard
    "MSFT",        # Microsoft
    "LMT",         # Lockheed Martin
    "KO" ,         # Coca-Cola
    "PLTR"  ,      #Palantir    
    "TTE.PA"       #Total
    ]
    
    print("Initializing Hierarchical Risk Parity Optimizer...")
    print("Weight constraint: Maximum 25% per asset")
    
    hrp = HRPOptimizer(tickers=tickers)
    
    hrp_weights = hrp.optimize()
    
    # Display results
    hrp.displayResults()
    
    # Plot comprehensive analysis
    hrp.plotResults()
    
    # Show final HRP weights with constraint verification
    max_weight_found = 0
    print("\nFINAL HRP WEIGHTS (Max weight for one security: 25%):")
    for ticker, weight in hrp_weights.items():
        print(f"{ticker}: {weight*100:.2f}%")
        max_weight_found = max(max_weight_found, weight)
    
    print(f"\nMaximum weight achieved: {max_weight_found*100:.2f}%")
    
    return hrp

if __name__ == '__main__':
    hrp_optimizer = main()