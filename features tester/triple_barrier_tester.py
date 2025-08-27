import pandas as pd
import numpy as np
from numba import jit

class TripleBarrierLabeling:
    """
    Classe pour implémenter la méthode Triple Barrier
    """
    
    def __init__(self, profit_target=0.03, stop_loss=0.02, max_hold_days=5):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
    
    @staticmethod
    @jit(nopython=True)
    def _find_first_barrier_hit(prices, entry_idx, profit_target, stop_loss, max_hold):
        """
        Fonction optimisée pour trouver la première barrière touchée
        """
        entry_price = prices[entry_idx]
        end_idx = min(entry_idx + max_hold, len(prices) - 1)
        
        for i in range(entry_idx + 1, end_idx + 1):
            current_price = prices[i]
            ret = (current_price - entry_price) / entry_price
            
            if ret >= profit_target:
                return 1, i  # Profit hit
            elif ret <= -stop_loss:
                return -1, i  # Stop loss hit
        
        return 0, end_idx  # Time barrier hit
    
    def create_labels(self, prices, entry_points=None, volatility_scaling=True):
        """
        Crée les labels Triple Barrier
        
        Parameters:
        -----------
        prices : pd.Series
            Série des prix
        entry_points : pd.Series, optional
            Points d'entrée spécifiques (si None, tous les points)
        volatility_scaling : bool
            Ajustement des barrières selon la volatilité
        
        Returns:
        --------
        pd.DataFrame : DataFrame avec labels et métadonnées
        """
        
        if entry_points is None:
            entry_points = pd.Series(True, index=prices.index)
        
        results = []
        prices_array = prices.values
        
        # Calcul de la volatilité roulante si scaling activé
        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std()
            vol = vol.fillna(vol.dropna().iloc[0])
        
        for i, (date, is_entry) in enumerate(entry_points.items()):
            if not is_entry or i >= len(prices) - 1:
                continue
                
            # Ajustement des barrières selon la volatilité
            if volatility_scaling:
                current_vol = vol.iloc[i]
                vol_adj = max(current_vol / 0.02, 0.5)  # Normalisation
                profit_adj = self.profit_target * vol_adj
                loss_adj = self.stop_loss * vol_adj
            else:
                profit_adj = self.profit_target
                loss_adj = self.stop_loss
            
            # Recherche de la première barrière
            label, exit_idx = self._find_first_barrier_hit(
                prices_array, i, profit_adj, loss_adj, self.max_hold_days
            )
            
            # Calcul des métriques du trade
            entry_price = prices_array[i]
            exit_price = prices_array[exit_idx]
            return_pct = (exit_price - entry_price) / entry_price
            hold_days = exit_idx - i
            
            results.append({
                'entry_date': date,
                'exit_date': prices.index[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': return_pct,
                'hold_days': hold_days,
                'label': label,
                'barrier_hit': ['Time', 'Profit', 'Loss'][label + 1],
                'vol_adjustment': vol_adj if volatility_scaling else 1.0
            })
        
        return pd.DataFrame(results)
    
    def analyze_performance(self, labels_df):
        """
        Analyse des performances de la stratégie de labelling
        """
        stats = {
            'total_trades': len(labels_df),
            'profit_trades': len(labels_df[labels_df['label'] == 1]),
            'loss_trades': len(labels_df[labels_df['label'] == -1]),
            'time_trades': len(labels_df[labels_df['label'] == 0]),
            'win_rate': len(labels_df[labels_df['label'] == 1]) / len(labels_df),
            'avg_return': labels_df['return'].mean(),
            'avg_hold_days': labels_df['hold_days'].mean(),
            'sharpe_ratio': labels_df['return'].mean() / labels_df['return'].std() * np.sqrt(252)
        }
        
        return stats

# Exemple d'utilisation avancée
def demonstrate_triple_barrier():
    """Démonstration complète de la méthode Triple Barrier"""
    
    # Génération de données synthétiques
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Prix avec tendance et volatilité variable
    returns = np.random.normal(0.001, 0.02, 252)
    returns[:60] *= 0.5  # Période de faible volatilité
    returns[60:120] *= 2.0  # Période de haute volatilité
    
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    # Application de la méthode
    tb_labeler = TripleBarrierLabeling(
        profit_target=0.03,
        stop_loss=0.02, 
        max_hold_days=10
    )
    
    # Création des labels avec et sans ajustement de volatilité
    labels_standard = tb_labeler.create_labels(prices, volatility_scaling=False)
    labels_vol_adj = tb_labeler.create_labels(prices, volatility_scaling=True)
    
    # Analyse comparative
    stats_standard = tb_labeler.analyze_performance(labels_standard)
    stats_vol_adj = tb_labeler.analyze_performance(labels_vol_adj)
    
    print("=== ANALYSE COMPARATIVE TRIPLE BARRIER ===")
    print("\nStandard (sans ajustement):")
    for key, value in stats_standard.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAvec ajustement de volatilité:")
    for key, value in stats_vol_adj.items():
        print(f"  {key}: {value:.4f}")
    
    return labels_standard, labels_vol_adj

# Exécution de la démonstration
labels_std, labels_vol = demonstrate_triple_barrier()
