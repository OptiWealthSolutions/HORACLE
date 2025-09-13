import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import uuid
from typing import Dict, List, Tuple, Optional

# Gestion des imports optionnels
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("⚠️ yfinance n'est pas installé. Utilisez: pip install yfinance")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ plotly n'est pas installé. Utilisez: pip install plotly")

# Configuration de la page
st.set_page_config(
    page_title="InvestTracker Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour une interface moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .positive { color: #00C851; }
    .negative { color: #ff4444; }
    .neutral { color: #33b5e5; }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fichiers pour sauvegarder les données
PORTFOLIO_FILE = "portfolio_positions.csv"
TRANSACTIONS_FILE = "portfolio_transactions.csv"

# Classes d'actifs disponibles
ASSET_CLASSES = {
    "Actions": "🏢",
    "ETF": "📊",
    "Crypto": "₿",
    "Obligations": "📋",
    "Matières Premières": "🥇",
    "Immobilier": "🏠",
    "Autre": "💼"
}

# Données de demo pour les prix (quand yfinance n'est pas disponible)
DEMO_PRICES = {
    "AAPL": 150.00,
    "GOOGL": 2500.00,
    "MSFT": 300.00,
    "TSLA": 800.00,
    "BTC-USD": 45000.00,
    "ETH-USD": 3000.00,
    "SPY": 450.00,
    "QQQ": 350.00
}

class PortfolioManager:
    def __init__(self):
        self.positions_df = self.load_positions()
        self.transactions_df = self.load_transactions()
    
    def load_positions(self) -> pd.DataFrame:
        """Charge les positions actuelles"""
        if os.path.exists(PORTFOLIO_FILE):
            try:
                df = pd.read_csv(PORTFOLIO_FILE)
                if 'date_creation' in df.columns:
                    df['date_creation'] = pd.to_datetime(df['date_creation'])
                return df
            except Exception as e:
                st.warning(f"Erreur lors du chargement des positions: {e}")
        return pd.DataFrame(columns=['id', 'symbole', 'nom', 'classe_actif', 'quantite', 'prix_moyen', 'date_creation'])
    
    def load_transactions(self) -> pd.DataFrame:
        """Charge l'historique des transactions"""
        if os.path.exists(TRANSACTIONS_FILE):
            try:
                df = pd.read_csv(TRANSACTIONS_FILE)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                st.warning(f"Erreur lors du chargement des transactions: {e}")
        return pd.DataFrame(columns=['id', 'position_id', 'type', 'quantite', 'prix', 'frais', 'date', 'notes'])
    
    def save_data(self):
        """Sauvegarde toutes les données"""
        try:
            self.positions_df.to_csv(PORTFOLIO_FILE, index=False)
            self.transactions_df.to_csv(TRANSACTIONS_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> float:
        """Récupère le prix actuel d'un symbole"""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    return data['Close'].iloc[-1]
            except Exception as e:
                st.warning(f"Erreur Yahoo Finance pour {symbol}: {e}")
        
        # Utiliser les prix de demo si yfinance n'est pas disponible
        return DEMO_PRICES.get(symbol.upper(), 100.0)
    
    def validate_symbol(self, symbol: str) -> bool:
        """Valide qu'un symbole existe"""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol.upper())
                data = ticker.history(period="1d")
                return not data.empty
            except:
                pass
        
        # Accepter les symboles de demo
        return symbol.upper() in DEMO_PRICES or len(symbol) <= 6
    
    def add_position(self, symbole: str, nom: str, classe_actif: str, quantite: float, prix: float, frais: float, date: datetime, notes: str = "") -> str:
        """Ajoute une nouvelle position ou met à jour une position existante"""
        position_id = None
        symbole = symbole.upper()
        
        # Vérifier si une position existe déjà pour ce symbole
        existing_position = self.positions_df[self.positions_df['symbole'] == symbole]
        
        if not existing_position.empty:
            # Position existe, mettre à jour
            position_id = existing_position.iloc[0]['id']
            idx = existing_position.index[0]
            
            # Calculer nouveau prix moyen et quantité
            old_quantite = self.positions_df.loc[idx, 'quantite']
            old_prix_moyen = self.positions_df.loc[idx, 'prix_moyen']
            
            nouvelle_quantite = old_quantite + quantite
            if nouvelle_quantite > 0:
                nouveau_prix_moyen = ((old_quantite * old_prix_moyen) + (quantite * prix)) / nouvelle_quantite
                self.positions_df.loc[idx, 'quantite'] = nouvelle_quantite
                self.positions_df.loc[idx, 'prix_moyen'] = nouveau_prix_moyen
            else:
                # Position fermée
                self.positions_df = self.positions_df.drop(idx).reset_index(drop=True)
        else:
            # Nouvelle position
            position_id = str(uuid.uuid4())
            new_position = pd.DataFrame({
                'id': [position_id],
                'symbole': [symbole],
                'nom': [nom],
                'classe_actif': [classe_actif],
                'quantite': [quantite],
                'prix_moyen': [prix],
                'date_creation': [date]
            })
            self.positions_df = pd.concat([self.positions_df, new_position], ignore_index=True)
        
        # Ajouter la transaction
        transaction = pd.DataFrame({
            'id': [str(uuid.uuid4())],
            'position_id': [position_id],
            'type': ['ACHAT'],
            'quantite': [quantite],
            'prix': [prix],
            'frais': [frais],
            'date': [date],
            'notes': [notes]
        })
        self.transactions_df = pd.concat([self.transactions_df, transaction], ignore_index=True)
        
        self.save_data()
        return position_id
    
    def sell_position(self, position_id: str, quantite: float, prix: float, frais: float, date: datetime, notes: str = ""):
        """Vend partiellement ou totalement une position"""
        position_idx = self.positions_df[self.positions_df['id'] == position_id].index
        
        if not position_idx.empty:
            idx = position_idx[0]
            current_quantite = self.positions_df.loc[idx, 'quantite']
            
            if quantite <= current_quantite:
                # Mettre à jour la quantité
                nouvelle_quantite = current_quantite - quantite
                
                if nouvelle_quantite > 0:
                    self.positions_df.loc[idx, 'quantite'] = nouvelle_quantite
                else:
                    # Position complètement vendue
                    self.positions_df = self.positions_df.drop(idx).reset_index(drop=True)
                
                # Ajouter la transaction de vente
                transaction = pd.DataFrame({
                    'id': [str(uuid.uuid4())],
                    'position_id': [position_id],
                    'type': ['VENTE'],
                    'quantite': [quantite],
                    'prix': [prix],
                    'frais': [frais],
                    'date': [date],
                    'notes': [notes]
                })
                self.transactions_df = pd.concat([self.transactions_df, transaction], ignore_index=True)
                self.save_data()
                return True
        return False
    
    def get_current_prices(self) -> Dict[str, float]:
        """Récupère les prix actuels de tous les symboles"""
        prices = {}
        if not self.positions_df.empty:
            symbols = self.positions_df['symbole'].unique()
            for symbol in symbols:
                prices[symbol] = self.get_current_price(symbol)
        return prices
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calcule toutes les métriques du portfolio"""
        if self.positions_df.empty:
            return {
                'total_invested': 0.0,
                'current_value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'daily_pnl': 0.0,
                'positions_count': 0,
                'total_fees': 0.0
            }
        
        # Prix actuels
        current_prices = self.get_current_prices()
        
        # Calculs pour positions ouvertes
        total_invested = 0.0
        current_value = 0.0
        
        for _, position in self.positions_df.iterrows():
            invested = position['quantite'] * position['prix_moyen']
            current = position['quantite'] * current_prices.get(position['symbole'], 0.0)
            total_invested += invested
            current_value += current
        
        # P&L non réalisé
        unrealized_pnl = current_value - total_invested
        
        # P&L réalisé et frais
        realized_pnl = 0.0
        total_fees = self.transactions_df['frais'].sum() if not self.transactions_df.empty else 0.0
        
        # Calculer le P&L réalisé
        closed_positions = self.calculate_realized_pnl()
        realized_pnl = sum(closed_positions.values())
        
        # Métriques globales
        total_pnl = unrealized_pnl + realized_pnl - total_fees
        total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
        
        return {
            'total_invested': total_invested,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'daily_pnl': 0.0,
            'positions_count': len(self.positions_df),
            'total_fees': total_fees
        }
    
    def calculate_realized_pnl(self) -> Dict[str, float]:
        """Calcule le P&L réalisé par position"""
        realized_pnl = {}
        
        if self.transactions_df.empty:
            return realized_pnl
        
        # Grouper les transactions par position
        for position_id in self.transactions_df['position_id'].unique():
            position_transactions = self.transactions_df[self.transactions_df['position_id'] == position_id]
            
            achats = position_transactions[position_transactions['type'] == 'ACHAT']
            ventes = position_transactions[position_transactions['type'] == 'VENTE']
            
            if not ventes.empty and not achats.empty:
                # Calculer le prix moyen d'achat
                total_quantite_achat = achats['quantite'].sum()
                prix_moyen_achat = (achats['quantite'] * achats['prix']).sum() / total_quantite_achat if total_quantite_achat > 0 else 0
                
                # Calculer le P&L des ventes
                for _, vente in ventes.iterrows():
                    pnl_vente = (vente['prix'] - prix_moyen_achat) * vente['quantite'] - vente['frais']
                    realized_pnl[position_id] = realized_pnl.get(position_id, 0) + pnl_vente
        
        return realized_pnl
    
    def get_portfolio_allocation(self) -> pd.DataFrame:
        """Retourne la répartition du portfolio par classe d'actif"""
        if self.positions_df.empty:
            return pd.DataFrame()
        
        current_prices = self.get_current_prices()
        
        allocation_data = []
        for _, position in self.positions_df.iterrows():
            value = position['quantite'] * current_prices.get(position['symbole'], 0.0)
            allocation_data.append({
                'classe_actif': position['classe_actif'],
                'valeur': value,
                'symbole': position['symbole']
            })
        
        df_allocation = pd.DataFrame(allocation_data)
        if not df_allocation.empty:
            return df_allocation.groupby('classe_actif')['valeur'].sum().reset_index()
        return pd.DataFrame()
    
    def get_detailed_positions(self) -> pd.DataFrame:
        """Retourne les positions détaillées avec métriques"""
        if self.positions_df.empty:
            return pd.DataFrame()
        
        current_prices = self.get_current_prices()
        detailed_positions = []
        
        for _, position in self.positions_df.iterrows():
            current_price = current_prices.get(position['symbole'], 0.0)
            invested = position['quantite'] * position['prix_moyen']
            current_value = position['quantite'] * current_price
            pnl_abs = current_value - invested
            pnl_pct = (pnl_abs / invested * 100) if invested > 0 else 0.0
            
            detailed_positions.append({
                'id': position['id'],
                'symbole': position['symbole'],
                'nom': position['nom'],
                'classe_actif': position['classe_actif'],
                'quantite': position['quantite'],
                'prix_moyen': position['prix_moyen'],
                'prix_actuel': current_price,
                'valeur_investie': invested,
                'valeur_actuelle': current_value,
                'pnl_absolu': pnl_abs,
                'pnl_pourcentage': pnl_pct,
                'date_creation': position['date_creation']
            })
        
        return pd.DataFrame(detailed_positions)

# Initialisation du gestionnaire de portfolio
@st.cache_resource
def get_portfolio_manager():
    return PortfolioManager()

# Interface utilisateur
st.markdown('<div class="main-header"><h1>📊 InvestTracker Pro</h1><p>Votre dashboard d\'investissement professionnel</p></div>', unsafe_allow_html=True)

# Vérification des dépendances
if not YFINANCE_AVAILABLE:
    st.markdown("""
    <div class="info-box">
        <h4>📦 Installation Requise</h4>
        <p>Pour utiliser toutes les fonctionnalités, installez les dépendances manquantes :</p>
        <code>pip install yfinance plotly</code>
        <p><em>En mode démo, des prix fictifs sont utilisés pour les symboles courants.</em></p>
    </div>
    """, unsafe_allow_html=True)

try:
    portfolio_manager = get_portfolio_manager()
except Exception as e:
    st.error(f"Erreur d'initialisation: {e}")
    st.stop()

# Sidebar pour les actions
with st.sidebar:
    st.markdown("## 🎯 Actions Rapides")
    
    if not YFINANCE_AVAILABLE:
        st.warning("Mode démo - Prix fictifs utilisés")
        st.markdown("**Symboles de demo disponibles:**")
        for symbol in list(DEMO_PRICES.keys())[:5]:
            st.markdown(f"• {symbol}")
    
    # Tabs pour organiser les fonctionnalités
    tab1, tab2, tab3 = st.tabs(["➕ Acheter", "💰 Vendre", "⚙️ Gérer"])
    
    with tab1:
        st.markdown("### Nouvelle Position")
        with st.form("add_position_form", clear_on_submit=True):
            symbole = st.text_input("Symbole", placeholder="ex: AAPL, BTC-USD", help="Symbole Yahoo Finance")
            nom = st.text_input("Nom", placeholder="ex: Apple Inc.")
            classe_actif = st.selectbox("Classe d'actif", list(ASSET_CLASSES.keys()))
            
            col1, col2 = st.columns(2)
            with col1:
                quantite = st.number_input("Quantité", min_value=0.000001, step=1.0, format="%.6f")
            with col2:
                prix = st.number_input("Prix unitaire", min_value=0.01, step=0.01)
            
            frais = st.number_input("Frais", min_value=0.0, step=0.01, help="Frais de courtage")
            date_achat = st.date_input("Date", value=datetime.now().date())
            notes = st.text_area("Notes", placeholder="Optionnel...")
            
            if st.form_submit_button("🚀 Acheter", use_container_width=True):
                if symbole and nom and quantite > 0 and prix > 0:
                    # Vérifier que le symbole existe
                    if portfolio_manager.validate_symbol(symbole):
                        try:
                            portfolio_manager.add_position(
                                symbole, nom, classe_actif, quantite, prix, frais,
                                pd.to_datetime(date_achat), notes
                            )
                            st.success(f"✅ Position {symbole} ajoutée!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de l'ajout: {e}")
                    else:
                        st.error(f"❌ Symbole {symbole} non reconnu")
                else:
                    st.error("❌ Veuillez remplir tous les champs obligatoires")
    
    with tab2:
        if not portfolio_manager.positions_df.empty:
            st.markdown("### Vendre Position")
            with st.form("sell_position_form", clear_on_submit=True):
                # Sélection de la position
                position_options = {}
                for _, pos in portfolio_manager.positions_df.iterrows():
                    label = f"{pos['symbole']} - {pos['nom']} ({pos['quantite']:.6f})"
                    position_options[label] = pos['id']
                
                selected_position_label = st.selectbox("Position à vendre", list(position_options.keys()))
                selected_position_id = position_options[selected_position_label]
                
                # Récupérer les détails de la position
                position_details = portfolio_manager.positions_df[
                    portfolio_manager.positions_df['id'] == selected_position_id
                ].iloc[0]
                
                max_quantite = position_details['quantite']
                
                col1, col2 = st.columns(2)
                with col1:
                    quantite_vente = st.number_input(
                        "Quantité à vendre", 
                        min_value=0.000001, 
                        max_value=float(max_quantite),
                        step=0.000001,
                        format="%.6f"
                    )
                with col2:
                    prix_vente = st.number_input("Prix de vente", min_value=0.01, step=0.01)
                
                frais_vente = st.number_input("Frais de vente", min_value=0.0, step=0.01)
                date_vente = st.date_input("Date de vente", value=datetime.now().date())
                notes_vente = st.text_area("Notes vente", placeholder="Optionnel...")
                
                if st.form_submit_button("💰 Vendre", use_container_width=True):
                    if quantite_vente > 0 and prix_vente > 0:
                        try:
                            success = portfolio_manager.sell_position(
                                selected_position_id, quantite_vente, prix_vente, frais_vente,
                                pd.to_datetime(date_vente), notes_vente
                            )
                            if success:
                                st.success(f"✅ Vente de {quantite_vente} {position_details['symbole']} effectuée!")
                                st.rerun()
                            else:
                                st.error("❌ Erreur lors de la vente")
                        except Exception as e:
                            st.error(f"Erreur: {e}")
                    else:
                        st.error("❌ Quantité et prix doivent être positifs")
        else:
            st.info("Aucune position à vendre")
    
    with tab3:
        st.markdown("### Gestion Portfolio")
        if st.button("🔄 Actualiser Prix", use_container_width=True):
            st.rerun()
        
        if st.button("📊 Exporter CSV", use_container_width=True):
            if not portfolio_manager.positions_df.empty:
                csv = portfolio_manager.positions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        if not portfolio_manager.positions_df.empty:
            st.markdown("#### 🗑️ Supprimer Position")
            position_to_delete = st.selectbox(
                "Position",
                options=portfolio_manager.positions_df['id'].tolist(),
                format_func=lambda x: portfolio_manager.positions_df[
                    portfolio_manager.positions_df['id'] == x
                ]['symbole'].iloc[0]
            )
            
            if st.button("❌ Supprimer", use_container_width=True):
                try:
                    portfolio_manager.positions_df = portfolio_manager.positions_df[
                        portfolio_manager.positions_df['id'] != position_to_delete
                    ].reset_index(drop=True)
                    portfolio_manager.save_data()
                    st.success("Position supprimée!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")

# Contenu principal
if portfolio_manager.positions_df.empty:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3>🎯 Commencez votre voyage d'investissement</h3>
        <p>Ajoutez votre première position via la barre latérale pour commencer à suivre vos investissements.</p>
        <p><em>Utilisez les symboles de demo si yfinance n'est pas installé : AAPL, GOOGL, MSFT, TSLA, BTC-USD</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Calcul des métriques
    with st.spinner("📊 Mise à jour des données..."):
        try:
            metrics = portfolio_manager.calculate_portfolio_metrics()
        except Exception as e:
            st.error(f"Erreur calcul des métriques: {e}")
            metrics = {
                'total_invested': 0.0, 'current_value': 0.0, 'unrealized_pnl': 0.0,
                'realized_pnl': 0.0, 'total_pnl': 0.0, 'total_return_pct': 0.0,
                'daily_pnl': 0.0, 'positions_count': 0, 'total_fees': 0.0
            }
    
    # Affichage des métriques principales
    st.markdown("### 📈 Vue d'ensemble du Portfolio")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Capital Investi",
            f"€{metrics['total_invested']:,.2f}",
            help="Montant total investi (hors frais)"
        )
    
    with col2:
        st.metric(
            "📊 Valeur Actuelle",
            f"€{metrics['current_value']:,.2f}",
            help="Valeur actuelle du portfolio"
        )
    
    with col3:
        pnl_color = "normal" if metrics['unrealized_pnl'] >= 0 else "inverse"
        st.metric(
            "📈 P&L Non Réalisé",
            f"€{metrics['unrealized_pnl']:,.2f}",
            delta=f"{metrics['total_return_pct']:.2f}%",
            delta_color=pnl_color,
            help="Plus-value ou moins-value latente"
        )
    
    with col4:
        realized_color = "normal" if metrics['realized_pnl'] >= 0 else "inverse"
        st.metric(
            "💸 P&L Réalisé",
            f"€{metrics['realized_pnl']:,.2f}",
            delta_color=realized_color,
            help="Plus-value ou moins-value réalisée"
        )
    
    with col5:
        st.metric(
            "🎯 Positions",
            f"{metrics['positions_count']}",
            help="Nombre de positions ouvertes"
        )
    
    # Deuxième ligne de métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pnl = metrics['total_pnl']
        total_color = "🟢" if total_pnl >= 0 else "🔴"
        st.metric(
            f"{total_color} P&L Total",
            f"€{total_pnl:,.2f}",
            help="P&L total (réalisé + non réalisé - frais)"
        )
    
    with col2:
        st.metric(
            "💳 Frais Totaux",
            f"€{metrics['total_fees']:,.2f}",
            help="Total des frais de transaction"
        )
    
    # Tableau des positions détaillées
    st.markdown("### 📋 Positions Détaillées")
    
    try:
        detailed_positions = portfolio_manager.get_detailed_positions()
        if not detailed_positions.empty:
            # Formatage pour l'affichage
            display_df = detailed_positions.copy()
            
            # Ajout des icônes de classe d'actif
            display_df['classe_actif'] = display_df['classe_actif'].apply(
                lambda x: f"{ASSET_CLASSES.get(x, '💼')} {x}"
            )
            
            # Configuration des colonnes
            format_config = {
                'symbole': st.column_config.TextColumn('Symbole', width=80),
                'nom': st.column_config.TextColumn('Nom', width=150),
                'classe_actif': st.column_config.TextColumn('Classe', width=120),
                'quantite': st.column_config.NumberColumn('Quantité', format="%.6f"),
                'prix_moyen': st.column_config.NumberColumn('Prix Moyen', format="€%.2f"),
                'prix_actuel': st.column_config.NumberColumn('Prix Actuel', format="€%.2f"),
                'valeur_investie': st.column_config.NumberColumn('Investi', format="€%.2f"),
                'valeur_actuelle': st.column_config.NumberColumn('Valeur', format="€%.2f"),
                'pnl_absolu': st.column_config.NumberColumn('P&L', format="€%.2f"),
                'pnl_pourcentage': st.column_config.NumberColumn('P&L %', format="%.2f%%"),
            }
            
            st.dataframe(
                display_df[['symbole', 'nom', 'classe_actif', 'quantite', 'prix_moyen', 
                           'prix_actuel', 'valeur_investie', 'valeur_actuelle', 
                           'pnl_absolu', 'pnl_pourcentage']],
                column_config=format_config,
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Erreur affichage positions: {e}")
    
    # Graphiques et analyses (seulement si Plotly est disponible)
    if PLOTLY_AVAILABLE:
        st.markdown("### 📊 Analyses Visuelles")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Répartition", "📈 Performance", "📋 Historique", "🔍 Analyse"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Répartition par classe d'actif
                try:
                    allocation_df = portfolio_manager.get_portfolio_allocation()
                    if not allocation_df.empty:
                        allocation_df['classe_icon'] = allocation_df['classe_actif'].apply(
                            lambda x: f"{ASSET_CLASSES.get(x, '💼')} {x}"
                        )
                        
                        fig_pie = px.pie(
                            allocation_df, 
                            values='valeur', 
                            names='classe_icon',
                            title="Répartition par Classe d'Actif",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(showlegend=True, height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur graphique répartition: {e}")
            
            with col2:
                # Top positions par valeur
                try:
                    if not detailed_positions.empty:
                        top_positions = detailed_positions.nlargest(5, 'valeur_actuelle')
                        fig_bar = px.bar(
                            top_positions,
                            x='valeur_actuelle',
                            y='symbole',
                            orientation='h',
                            title="Top 5 Positions par Valeur",
                            color='pnl_pourcentage',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_bar.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur graphique top positions: {e}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L par position
                try:
                    if not detailed_positions.empty:
                        fig_pnl = px.bar(
                            detailed_positions,
                            x='symbole',
                            y='pnl_absolu',
                            title="P&L par Position",
                            color='pnl_absolu',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_pnl.update_layout(height=400)
                        st.plotly_chart(fig_pnl, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur graphique P&L: {e}")
            
            with col2:
                # Performance en pourcentage
                try:
                    if not detailed_positions.empty:
                        fig_perf = px.scatter(
                            detailed_positions,
                            x='valeur_investie',
                            y='pnl_pourcentage',
                            size='valeur_actuelle',
                            color='classe_actif',
                            hover_name='symbole',
                            title="Performance vs Montant Investi",
                            labels={'valeur_investie': 'Montant Investi (€)', 'pnl_pourcentage': 'Performance (%)'}
                        )
                        fig_perf.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_perf.update_layout(height=400)
                        st.plotly_chart(fig_perf, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur graphique performance: {e}")
        
        with tab3:
            # Historique des transactions
            if not portfolio_manager.transactions_df.empty:
                st.markdown("#### 📋 Historique des Transactions")
                
                try:
                    transactions_display = portfolio_manager.transactions_df.copy()
                    transactions_display = transactions_display.sort_values('date', ascending=False)
                    
                    # Formatage pour l'affichage
                    transactions_display['montant'] = transactions_display['quantite'] * transactions_display['prix']
                    transactions_display['type_icon'] = transactions_display['type'].apply(
                        lambda x: "🟢 ACHAT" if x == "ACHAT" else "🔴 VENTE"
                    )
                    
                    # Joindre avec les positions pour avoir le symbole
                    if not portfolio_manager.positions_df.empty:
                        transactions_with_symbol = transactions_display.merge(
                            portfolio_manager.positions_df[['id', 'symbole']], 
                            left_on='position_id', 
                            right_on='id',
                            how='left',
                            suffixes=('', '_pos')
                        )
                    else:
                        transactions_with_symbol = transactions_display.copy()
                        transactions_with_symbol['symbole'] = 'N/A'
                    
                    # Configuration des colonnes
                    transaction_config = {
                        'date': st.column_config.DateColumn('Date'),
                        'type_icon': st.column_config.TextColumn('Type', width=100),
                        'symbole': st.column_config.TextColumn('Symbole', width=80),
                        'quantite': st.column_config.NumberColumn('Quantité', format="%.6f"),
                        'prix': st.column_config.NumberColumn('Prix', format="€%.2f"),
                        'montant': st.column_config.NumberColumn('Montant', format="€%.2f"),
                        'frais': st.column_config.NumberColumn('Frais', format="€%.2f"),
                        'notes': st.column_config.TextColumn('Notes', width=200)
                    }
                    
                    st.dataframe(
                        transactions_with_symbol[['date', 'type_icon', 'symbole', 'quantite', 
                                                'prix', 'montant', 'frais', 'notes']],
                        column_config=transaction_config,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Graphique des transactions dans le temps
                    monthly_transactions = transactions_with_symbol.copy()
                    monthly_transactions['mois'] = monthly_transactions['date'].dt.to_period('M').astype(str)
                    monthly_summary = monthly_transactions.groupby(['mois', 'type']).agg({
                        'montant': 'sum'
                    }).reset_index()
                    
                    if not monthly_summary.empty:
                        fig_transactions = px.bar(
                            monthly_summary,
                            x='mois',
                            y='montant',
                            color='type',
                            title="Évolution des Transactions par Mois",
                            color_discrete_map={'ACHAT': '#00C851', 'VENTE': '#ff4444'}
                        )
                        fig_transactions.update_layout(height=400)
                        st.plotly_chart(fig_transactions, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur historique transactions: {e}")
            else:
                st.info("Aucune transaction enregistrée")
        
        with tab4:
            # Analyses avancées
            st.markdown("#### 🔍 Analyses Avancées")
            
            if not detailed_positions.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 📊 Statistiques du Portfolio")
                    
                    try:
                        # Calculs statistiques
                        total_positions = len(detailed_positions)
                        positions_positives = len(detailed_positions[detailed_positions['pnl_absolu'] > 0])
                        positions_negatives = len(detailed_positions[detailed_positions['pnl_absolu'] < 0])
                        taux_reussite = (positions_positives / total_positions * 100) if total_positions > 0 else 0
                        
                        meilleure_perf = detailed_positions.loc[detailed_positions['pnl_pourcentage'].idxmax()]
                        pire_perf = detailed_positions.loc[detailed_positions['pnl_pourcentage'].idxmin()]
                        
                        stats_data = {
                            "Métrique": [
                                "Positions Gagnantes",
                                "Positions Perdantes", 
                                "Taux de Réussite",
                                "Meilleure Performance",
                                "Pire Performance",
                                "Performance Moyenne",
                                "Volatilité Portfolio"
                            ],
                            "Valeur": [
                                f"{positions_positives}/{total_positions}",
                                f"{positions_negatives}/{total_positions}",
                                f"{taux_reussite:.1f}%",
                                f"{meilleure_perf['symbole']} ({meilleure_perf['pnl_pourcentage']:.1f}%)",
                                f"{pire_perf['symbole']} ({pire_perf['pnl_pourcentage']:.1f}%)",
                                f"{detailed_positions['pnl_pourcentage'].mean():.1f}%",
                                f"{detailed_positions['pnl_pourcentage'].std():.1f}%"
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur calcul statistiques: {e}")
                
                with col2:
                    st.markdown("##### 🎯 Répartition des Performances")
                    
                    try:
                        # Histogramme des performances
                        fig_hist = px.histogram(
                            detailed_positions,
                            x='pnl_pourcentage',
                            nbins=20,
                            title="Distribution des Performances (%)",
                            color_discrete_sequence=['#667eea']
                        )
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Seuil de Rentabilité")
                        fig_hist.update_layout(height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur histogramme: {e}")
            
            # Matrice de corrélation (seulement si yfinance est disponible)
            if YFINANCE_AVAILABLE and not detailed_positions.empty:
                st.markdown("##### 🔗 Analyse de Corrélation")
                
                symbols_for_correlation = detailed_positions['symbole'].head(5).tolist()
                
                if len(symbols_for_correlation) > 1:
                    try:
                        correlation_data = {}
                        
                        for symbol in symbols_for_correlation:
                            try:
                                ticker = yf.Ticker(symbol)
                                hist_data = ticker.history(period="6mo")
                                if not hist_data.empty:
                                    correlation_data[symbol] = hist_data['Close'].pct_change().dropna()
                            except:
                                continue
                        
                        if len(correlation_data) > 1:
                            # Aligner les données sur les mêmes dates
                            df_corr = pd.DataFrame(correlation_data)
                            df_corr = df_corr.dropna()
                            
                            if not df_corr.empty and len(df_corr.columns) > 1:
                                correlation_matrix = df_corr.corr()
                                
                                fig_corr = px.imshow(
                                    correlation_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    title="Matrice de Corrélation (6 mois)",
                                    color_continuous_scale='RdBu'
                                )
                                fig_corr.update_layout(height=400)
                                st.plotly_chart(fig_corr, use_container_width=True)
                            else:
                                st.info("Données insuffisantes pour calculer les corrélations")
                        else:
                            st.info("Au moins 2 positions nécessaires pour l'analyse de corrélation")
                    except Exception as e:
                        st.error(f"Erreur analyse corrélation: {e}")
        
        # Graphique d'évolution historique (seulement si yfinance est disponible)
        if YFINANCE_AVAILABLE and not detailed_positions.empty:
            st.markdown("### 📈 Évolution Historique d'une Position")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_symbol = st.selectbox(
                    "Sélectionner une position",
                    detailed_positions['symbole'].tolist(),
                    key="symbol_selector"
                )
            
            with col2:
                period = st.selectbox(
                    "Période",
                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    index=3,
                    key="period_selector"
                )
            
            if selected_symbol:
                try:
                    ticker = yf.Ticker(selected_symbol)
                    historical_data = ticker.history(period=period)
                    
                    if not historical_data.empty:
                        # Informations sur la position sélectionnée
                        position_info = detailed_positions[detailed_positions['symbole'] == selected_symbol].iloc[0]
                        
                        # Création des sous-graphiques
                        fig_combined = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f'Prix de {selected_symbol}', 'Volume'),
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Ajout des traces au graphique combiné
                        fig_combined.add_trace(
                            go.Scatter(
                                x=historical_data.index,
                                y=historical_data['Close'],
                                mode='lines',
                                name='Prix',
                                line=dict(color='#667eea', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        fig_combined.add_trace(
                            go.Bar(
                                x=historical_data.index,
                                y=historical_data['Volume'],
                                name='Volume',
                                marker_color='rgba(102, 126, 234, 0.3)'
                            ),
                            row=2, col=1
                        )
                        
                        # Lignes de référence
                        fig_combined.add_hline(
                            y=position_info['prix_moyen'],
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"Prix moyen: €{position_info['prix_moyen']:.2f}",
                            row=1
                        )
                        
                        fig_combined.update_layout(
                            title=f"Évolution de {selected_symbol} - {period.upper()}",
                            height=600,
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        fig_combined.update_xaxes(title_text="Date", row=2, col=1)
                        fig_combined.update_yaxes(title_text="Prix (€)", row=1, col=1)
                        fig_combined.update_yaxes(title_text="Volume", row=2, col=1)
                        
                        st.plotly_chart(fig_combined, use_container_width=True)
                        
                        # Statistiques rapides
                        col1, col2, col3, col4 = st.columns(4)
                        
                        current_price = historical_data['Close'].iloc[-1]
                        price_change = ((current_price - historical_data['Close'].iloc[0]) / historical_data['Close'].iloc[0] * 100)
                        volatility = historical_data['Close'].pct_change().std() * np.sqrt(252) * 100  # Volatilité annualisée
                        max_price = historical_data['High'].max()
                        min_price = historical_data['Low'].min()
                        
                        with col1:
                            st.metric("📈 Variation Période", f"{price_change:.1f}%")
                        
                        with col2:
                            st.metric("📊 Volatilité Ann.", f"{volatility:.1f}%")
                        
                        with col3:
                            st.metric("⬆️ Plus Haut", f"€{max_price:.2f}")
                        
                        with col4:
                            st.metric("⬇️ Plus Bas", f"€{min_price:.2f}")
                        
                    else:
                        st.error(f"Impossible de récupérer les données pour {selected_symbol}")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des données: {str(e)}")
    else:
        # Interface simplifiée sans Plotly
        st.markdown("### 📊 Analyses Simples")
        
        if not detailed_positions.empty:
            # Statistiques de base
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positions_positives = len(detailed_positions[detailed_positions['pnl_absolu'] > 0])
                total_positions = len(detailed_positions)
                st.metric("🟢 Positions Gagnantes", f"{positions_positives}/{total_positions}")
            
            with col2:
                meilleure_perf = detailed_positions['pnl_pourcentage'].max()
                st.metric("📈 Meilleure Performance", f"{meilleure_perf:.1f}%")
            
            with col3:
                perf_moyenne = detailed_positions['pnl_pourcentage'].mean()
                st.metric("📊 Performance Moyenne", f"{perf_moyenne:.1f}%")
            
            # Répartition par classe (tableau simple)
            st.markdown("#### Répartition par Classe d'Actif")
            allocation_df = portfolio_manager.get_portfolio_allocation()
            if not allocation_df.empty:
                allocation_df['pourcentage'] = (allocation_df['valeur'] / allocation_df['valeur'].sum() * 100).round(1)
                allocation_df['valeur_formatee'] = allocation_df['valeur'].apply(lambda x: f"€{x:,.2f}")
                allocation_df['pourcentage_formatee'] = allocation_df['pourcentage'].apply(lambda x: f"{x}%")
                
                st.dataframe(
                    allocation_df[['classe_actif', 'valeur_formatee', 'pourcentage_formatee']],
                    column_config={
                        'classe_actif': 'Classe d\'Actif',
                        'valeur_formatee': 'Valeur',
                        'pourcentage_formatee': 'Pourcentage'
                    },
                    hide_index=True,
                    use_container_width=True
                )

# Footer avec informations système
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("💡 **InvestTracker Pro** - Dashboard professionnel")

with col2:
    if os.path.exists(PORTFOLIO_FILE):
        file_size = os.path.getsize(PORTFOLIO_FILE)
        st.markdown(f"💾 **Données**: {file_size} bytes sauvegardés")
    else:
        st.markdown("💾 **Données**: Aucune sauvegarde")

with col3:
    if st.button("ℹ️ À propos"):
        st.info(f"""
        **InvestTracker Pro v2.0**
        
        ✅ Gestion complète des positions
        ✅ Suivi des P&L réalisés et non réalisés
        ✅ Calculs financiers précis
        ✅ Interface moderne et responsive
        ✅ Sauvegarde automatique locale
        
        **Status des modules:**
        • yfinance: {'✅ Disponible' if YFINANCE_AVAILABLE else '❌ Non installé'}
        • plotly: {'✅ Disponible' if PLOTLY_AVAILABLE else '❌ Non installé'}
        
        **Pour installer les dépendances:**
        ```
        pip install yfinance plotly
        ```
        """)

# Instructions d'installation si nécessaire
if not YFINANCE_AVAILABLE or not PLOTLY_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Installation")
    st.sidebar.markdown("Pour activer toutes les fonctionnalités :")
    st.sidebar.code("pip install yfinance plotly", language="bash")