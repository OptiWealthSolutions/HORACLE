import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Tuple

# Configuration avancée de la page avec thème sombre
st.set_page_config(
    page_title="🚀 OptiWealth Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "OptiWealth Pro - Dashboard d'investissement professionnel"
    }
)

# CSS personnalisé pour une interface premium
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header personnalisé */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Métriques cards personnalisées */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2D1B69 0%, #11998e 100%);
    }
    
    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error alerts */
    .success-alert {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    .error-alert {
        background: linear-gradient(45deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration avancée
CSV_FILE = "portfolio_data.csv"
HISTORICAL_DATA_FILE = "historical_performance.csv"

# Classes d'actifs avec emojis
ASSET_CLASSES = {
    "🏢 Actions": "Actions",
    "📊 ETF": "ETF", 
    "₿ Crypto": "Crypto",
    "🏛️ Obligations": "Obligations",
    "🥇 Matières Premières": "Matières Premières",
    "🌐 Forex": "Forex",
    "🏠 Immobilier": "Immobilier",
    "💎 Autre": "Autre"
}

# Zones géographiques
GEOGRAPHICAL_ZONES = [
    "🇺🇸 États-Unis", "🇪🇺 Europe", "🇨🇳 Asie", "🌍 Émergents", "🌏 Global", "🇫🇷 France", "Autre"
]

class PortfolioAnalyzer:
    """Classe pour analyser les performances du portfolio"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcule le ratio de Sharpe"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns.mean() - risk_free_rate/252
        return (excess_returns / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
    
    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calcule le maximum drawdown"""
        if len(values) < 2:
            return 0.0
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calcule la volatilité annualisée"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(252)

def load_portfolio_data() -> pd.DataFrame:
    """Charge les données du portfolio avec gestion des erreurs avancée"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            df['date_achat'] = pd.to_datetime(df['date_achat'])
            
            # Migration automatique des colonnes
            required_columns = ['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais', 'zone_geo', 'secteur']
            for col in required_columns:
                if col not in df.columns:
                    if col in ['frais', 'montant']:
                        df[col] = 0.0
                    else:
                        df[col] = ""
            
            return df
        else:
            return pd.DataFrame(columns=required_columns)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais', 'zone_geo', 'secteur'])

def save_portfolio_data(df: pd.DataFrame):
    """Sauvegarde sécurisée des données"""
    try:
        # Backup de sécurité
        if os.path.exists(CSV_FILE):
            backup_file = f"{CSV_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(CSV_FILE, backup_file)
        
        df.to_csv(CSV_FILE, index=False)
        
        # Nettoyer les anciens backups (garder seulement les 5 derniers)
        backup_files = [f for f in os.listdir('.') if f.startswith(f"{CSV_FILE}.backup_")]
        if len(backup_files) > 5:
            backup_files.sort(reverse=True)
            for old_backup in backup_files[5:]:
                os.remove(old_backup)
                
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_current_price(symbol: str) -> float:
    """Récupère le prix actuel avec cache et gestion d'erreurs"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def get_ticker_info(symbol: str) -> Dict:
    """Récupère les informations détaillées d'un ticker"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'longName': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'country': info.get('country', ''),
            'marketCap': info.get('marketCap', 0),
            'beta': info.get('beta', 0),
            'dividendYield': info.get('dividendYield', 0)
        }
    except:
        return {}

def get_historical_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Récupère les données historiques"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except:
        return pd.DataFrame()

def create_advanced_metrics_display(df_portfolio: pd.DataFrame):
    """Crée un affichage avancé des métriques"""
    
    if df_portfolio.empty:
        st.info("🚀 Commencez votre parcours d'investissement en ajoutant votre première position!")
        return
    
    # Séparer positions ouvertes et ventes
    df_ouvertes = df_portfolio[df_portfolio['montant'] > 0].copy()
    df_ventes = df_portfolio[df_portfolio['montant'] < 0].copy()
    
    # Récupération des prix actuels avec barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_prices = []
    for i, symbol in enumerate(df_portfolio['symbole']):
        status_text.text(f'Récupération des prix... {symbol} ({i+1}/{len(df_portfolio)})')
        price = get_current_price(symbol)
        current_prices.append(price if price is not None else 0)
        progress_bar.progress((i + 1) / len(df_portfolio))
    
    df_portfolio['prix_actuel'] = current_prices
    progress_bar.empty()
    status_text.empty()
    
    # Calculs pour positions ouvertes
    if not df_ouvertes.empty:
        df_ouvertes['prix_actuel'] = df_portfolio.loc[df_ouvertes.index, 'prix_actuel']
        df_ouvertes['valeur_achat'] = df_ouvertes['montant'] + df_ouvertes['frais']
        df_ouvertes['valeur_actuelle'] = df_ouvertes['montant'] * (df_ouvertes['prix_actuel'] / df_ouvertes['prix_achat'])
        df_ouvertes['pnl_absolu'] = df_ouvertes['valeur_actuelle'] - df_ouvertes['valeur_achat']
        df_ouvertes['pnl_pct'] = np.where(
            df_ouvertes['valeur_achat'] != 0,
            (df_ouvertes['pnl_absolu'] / df_ouvertes['valeur_achat'] * 100),
            0.0
        )
    
    # Calculs pour ventes
    if not df_ventes.empty:
        df_ventes['valeur_achat'] = 0.0
        df_ventes['valeur_actuelle'] = 0.0
        df_ventes['pnl_absolu'] = -df_ventes['montant'] * (df_ventes['prix_achat'] - df_ventes['prix_achat']) - df_ventes['frais']
        df_ventes['pnl_pct'] = 0.0
    
    # Métriques globales
    capital_investi = df_ouvertes['valeur_achat'].sum() if not df_ouvertes.empty else 0
    valeur_actuelle = df_ouvertes['valeur_actuelle'].sum() if not df_ouvertes.empty else 0
    pnl_flottant = df_ouvertes['pnl_absolu'].sum() if not df_ouvertes.empty else 0
    pnl_realise = df_ventes['pnl_absolu'].sum() if not df_ventes.empty else 0
    pnl_total = pnl_flottant + pnl_realise
    rendement_total = (pnl_total / capital_investi * 100) if capital_investi > 0 else 0
    
    # Header premium avec dégradé
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🚀 OptiWealth Pro Dashboard</h1>
        <p class="header-subtitle">Tableau de bord professionnel pour vos investissements</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales avec design premium
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "normal" if capital_investi >= 0 else "inverse"
        st.metric("💰 Capital Investi", f"{capital_investi:,.0f} €", 
                 delta=f"{len(df_ouvertes)} positions", delta_color=delta_color)
    
    with col2:
        delta_color = "normal" if valeur_actuelle >= capital_investi else "inverse"
        st.metric("📈 Valeur Actuelle", f"{valeur_actuelle:,.0f} €",
                 delta=f"{((valeur_actuelle/capital_investi-1)*100):+.1f}%" if capital_investi > 0 else None,
                 delta_color=delta_color)
    
    with col3:
        delta_color = "normal" if pnl_flottant >= 0 else "inverse"
        st.metric("📊 P&L Flottant", f"{pnl_flottant:,.0f} €", 
                 delta=f"{(pnl_flottant/capital_investi*100):+.1f}%" if capital_investi > 0 else None,
                 delta_color=delta_color)
    
    with col4:
        delta_color = "normal" if pnl_realise >= 0 else "inverse"
        st.metric("💸 P&L Réalisé", f"{pnl_realise:,.0f} €",
                 delta=f"{len(df_ventes)} ventes" if len(df_ventes) > 0 else "0 ventes",
                 delta_color=delta_color)
    
    with col5:
        delta_color = "normal" if pnl_total >= 0 else "inverse"
        st.metric("🎯 Performance Totale", f"{pnl_total:,.0f} €",
                 delta=f"{rendement_total:+.1f}%", delta_color=delta_color)
    
    return df_portfolio, df_ouvertes, df_ventes

def create_advanced_visualizations(df_portfolio: pd.DataFrame, df_ouvertes: pd.DataFrame):
    """Crée des visualisations avancées et interactives"""
    
    st.header("📊 Analytics Avancés")
    
    # Tabs pour organiser les visualisations
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Allocation", "📈 Performance", "🔍 Analyse", "📅 Historique"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition par classe d'actif avec design moderne
            if not df_ouvertes.empty:
                df_allocation = df_ouvertes.groupby('classe_actif')['valeur_actuelle'].sum().reset_index()
                fig_pie = px.pie(df_allocation, values='valeur_actuelle', names='classe_actif',
                               title="🎯 Allocation par Classe d'Actif",
                               color_discrete_sequence=px.colors.qualitative.Set3,
                               hover_data=['valeur_actuelle'])
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    font_family="Inter",
                    title_font_size=18,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Répartition géographique
            if not df_ouvertes.empty and 'zone_geo' in df_ouvertes.columns:
                df_geo = df_ouvertes.groupby('zone_geo')['valeur_actuelle'].sum().reset_index()
                if not df_geo.empty and df_geo['zone_geo'].notna().any():
                    fig_geo = px.sunburst(df_geo, path=['zone_geo'], values='valeur_actuelle',
                                         title="🌍 Répartition Géographique",
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_geo.update_layout(font_family="Inter", title_font_size=18)
                    st.plotly_chart(fig_geo, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance par position
            if not df_ouvertes.empty:
                fig_perf = px.bar(df_ouvertes, x='symbole', y='pnl_pct',
                                 title="📊 Performance par Position (%)",
                                 color='pnl_pct',
                                 color_continuous_scale=['red', 'yellow', 'green'],
                                 hover_data=['nom', 'valeur_actuelle'])
                fig_perf.update_layout(
                    font_family="Inter",
                    title_font_size=18,
                    xaxis_title="Positions",
                    yaxis_title="Rendement (%)",
                    showlegend=False
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            # Top/Flop performers
            if not df_ouvertes.empty:
                top_performers = df_ouvertes.nlargest(5, 'pnl_pct')[['symbole', 'nom', 'pnl_pct']]
                bottom_performers = df_ouvertes.nsmallest(5, 'pnl_pct')[['symbole', 'nom', 'pnl_pct']]
                
                st.subheader("🏆 Top Performers")
                for _, row in top_performers.iterrows():
                    st.success(f"**{row['symbole']}** - {row['nom']}: {row['pnl_pct']:+.1f}%")
                
                st.subheader("⚠️ Attention Requise")
                for _, row in bottom_performers.iterrows():
                    st.error(f"**{row['symbole']}** - {row['nom']}: {row['pnl_pct']:+.1f}%")
    
    with tab3:
        # Analyse de risque avancée
        if not df_ouvertes.empty:
            analyzer = PortfolioAnalyzer(df_ouvertes)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Métriques de Risque")
                
                # Calcul de la volatilité simulée
                returns = df_ouvertes['pnl_pct'] / 100
                volatilite = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                sharpe = analyzer.calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
                
                st.metric("📈 Volatilité Annuelle", f"{volatilite:.1%}")
                st.metric("⚖️ Ratio de Sharpe", f"{sharpe:.2f}")
                
                # Jauge de risque
                risk_level = min(volatilite * 100, 100)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_level,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Niveau de Risque"},
                    delta={'reference': 20},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 20], 'color': "lightgreen"},
                                    {'range': [20, 50], 'color': "yellow"},
                                    {'range': [50, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 80}}))
                fig_gauge.update_layout(height=300, font_family="Inter")
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Recommandations")
                
                # Analyse et recommandations automatiques
                total_value = df_ouvertes['valeur_actuelle'].sum()
                largest_position = df_ouvertes.loc[df_ouvertes['valeur_actuelle'].idxmax()]
                concentration = (largest_position['valeur_actuelle'] / total_value) * 100
                
                if concentration > 30:
                    st.warning(f"⚠️ **Concentration élevée**: {largest_position['symbole']} représente {concentration:.1f}% du portfolio")
                    st.info("💡 Considérez diversifier davantage vos positions")
                
                # Secteurs sur-représentés
                if 'secteur' in df_ouvertes.columns:
                    sector_allocation = df_ouvertes.groupby('secteur')['valeur_actuelle'].sum()
                    max_sector = sector_allocation.max() / total_value * 100
                    if max_sector > 40:
                        st.warning(f"📊 Secteur sur-représenté: {max_sector:.1f}%")
                
                # Performance vs benchmark simulé
                portfolio_return = df_ouvertes['pnl_pct'].mean()
                if portfolio_return > 10:
                    st.success("🚀 Performance excellente!")
                elif portfolio_return > 5:
                    st.info("📈 Performance satisfaisante")
                else:
                    st.warning("🔍 Performance à améliorer")
    
    with tab4:
        # Graphique d'évolution historique avancé
        st.subheader("📈 Évolution Historique Avancée")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            selected_symbols = st.multiselect(
                "Sélectionner les positions",
                options=df_portfolio['symbole'].unique(),
                default=df_portfolio['symbole'].unique()[:3] if len(df_portfolio) >= 3 else df_portfolio['symbole'].unique()
            )
            
            period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            
            show_volume = st.checkbox("Afficher le volume", value=False)
            show_ma = st.checkbox("Moyennes mobiles", value=True)
        
        with col1:
            if selected_symbols:
                fig = make_subplots(
                    rows=2 if show_volume else 1,
                    cols=1,
                    shared_xaxis=True,
                    vertical_spacing=0.1,
                    subplot_titles=['Prix', 'Volume'] if show_volume else ['Prix'],
                    row_heights=[0.7, 0.3] if show_volume else [1]
                )
                
                colors = px.colors.qualitative.Set1
                
                for i, symbol in enumerate(selected_symbols):
                    historical_data = get_historical_data(symbol, period)
                    if not historical_data.empty:
                        color = colors[i % len(colors)]
                        
                        # Prix
                        fig.add_trace(
                            go.Scatter(
                                x=historical_data.index,
                                y=historical_data['Close'],
                                mode='lines',
                                name=f'{symbol}',
                                line=dict(color=color, width=2),
                                hovertemplate=f'<b>{symbol}</b><br>Prix: %{{y:.2f}}€<br>Date: %{{x}}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        
                        # Moyennes mobiles
                        if show_ma and len(historical_data) > 20:
                            ma20 = historical_data['Close'].rolling(20).mean()
                            ma50 = historical_data['Close'].rolling(50).mean()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=historical_data.index,
                                    y=ma20,
                                    mode='lines',
                                    name=f'{symbol} MA20',
                                    line=dict(color=color, width=1, dash='dash'),
                                    opacity=0.7
                                ),
                                row=1, col=1
                            )
                            
                            if len(historical_data) > 50:
                                fig.add_trace(
                                    go.Scatter(
                                        x=historical_data.index,
                                        y=ma50,
                                        mode='lines',
                                        name=f'{symbol} MA50',
                                        line=dict(color=color, width=1, dash='dot'),
                                        opacity=0.5
                                    ),
                                    row=1, col=1
                                )
                        
                        # Volume
                        if show_volume and 'Volume' in historical_data.columns:
                            fig.add_trace(
                                go.Bar(
                                    x=historical_data.index,
                                    y=historical_data['Volume'],
                                    name=f'{symbol} Volume',
                                    marker_color=color,
                                    opacity=0.6
                                ),
                                row=2, col=1
                            )
                        
                        # Prix d'achat
                        purchase_price = df_portfolio[df_portfolio['symbole'] == symbol]['prix_achat'].iloc[0]
                        fig.add_hline(
                            y=purchase_price,
                            line_dash="dash",
                            line_color=color,
                            annotation_text=f"{symbol}: {purchase_price:.2f}€",
                            row=1, col=1
                        )
                
                fig.update_layout(
                    title=f"Évolution des Positions - {period}",
                    font_family="Inter",
                    title_font_size=18,
                    height=600 if show_volume else 400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                )
                
                fig.update_xaxis(title_text="Date", row=2 if show_volume else 1, col=1)
                fig.update_yaxis(title_text="Prix (€)", row=1, col=1)
                if show_volume:
                    fig.update_yaxis(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)

def create_portfolio_table(df_portfolio: pd.DataFrame):
    """Crée un tableau de portfolio avancé et interactif"""
    
    st.header("📋 Portfolio Détaillé")
    
    if df_portfolio.empty:
        st.info("Aucune position dans votre portfolio.")
        return
    
    # Séparer positions ouvertes et fermées
    df_ouvertes = df_portfolio[df_portfolio['montant'] > 0].copy()
    df_fermees = df_portfolio[df_portfolio['montant'] < 0].copy()
    
    # Filtres interactifs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        classe_filter = st.multiselect(
            "Filtrer par classe d'actif",
            options=df_portfolio['classe_actif'].unique(),
            default=df_portfolio['classe_actif'].unique()
        )
    
    with col2:
        show_only_profitable = st.checkbox("Positions rentables uniquement", value=False)
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            options=['Symbole', 'Performance %', 'Valeur', 'P&L Absolu'],
            index=2
        )
    
    # Application des filtres
    df_filtered = df_ouvertes[df_ouvertes['classe_actif'].isin(classe_filter)] if classe_filter else df_ouvertes
    
    if show_only_profitable and not df_filtered.empty:
        df_filtered = df_filtered[df_filtered['pnl_pct'] > 0]
    
    # Tri
    if not df_filtered.empty:
        sort_mapping = {
            'Symbole': 'symbole',
            'Performance %': 'pnl_pct',
            'Valeur': 'valeur_actuelle',
            'P&L Absolu': 'pnl_absolu'
        }
        df_filtered = df_filtered.sort_values(sort_mapping[sort_by], ascending=False)
    
    # Formatage pour l'affichage
    if not df_filtered.empty:
        df_display = df_filtered.copy()
        df_display['prix_achat_fmt'] = df_display['prix_achat'].apply(lambda x: f"{x:.2f} €")
        df_display['prix_actuel_fmt'] = df_display['prix_actuel'].apply(lambda x: f"{x:.2f} €")
        df_display['valeur_achat_fmt'] = df_display['valeur_achat'].apply(lambda x: f"{x:,.0f} €")
        df_display['valeur_actuelle_fmt'] = df_display['valeur_actuelle'].apply(lambda x: f"{x:,.0f} €")
        df_display['pnl_absolu_fmt'] = df_display['pnl_absolu'].apply(lambda x: f"{x:+,.0f} €")
        df_display['pnl_pct_fmt'] = df_display['pnl_pct'].apply(lambda x: f"{x:+.1f}%")
        df_display['frais_fmt'] = df_display['frais'].apply(lambda x: f"{x:.2f} €")
        
        # Colonnes à afficher
        columns_display = {
            'symbole': 'Symbole',
            'nom': 'Nom',
            'classe_actif': 'Classe',
            'montant': 'Quantité/Montant',
            'prix_achat_fmt': 'Prix Achat',
            'prix_actuel_fmt': 'Prix Actuel',
            'valeur_actuelle_fmt': 'Valeur Actuelle',
            'pnl_absolu_fmt': 'P&L (€)',
            'pnl_pct_fmt': 'P&L (%)',
            'frais_fmt': 'Frais',
            'date_achat': 'Date Achat'
        }
        
        # Configuration des couleurs pour les colonnes P&L
        def color_pnl(val):
            if 'P&L' in val.name:
                if '+' in str(val):
                    return ['background-color: #d4edda; color: #155724'] * len(val)
                elif '-' in str(val):
                    return ['background-color: #f8d7da; color: #721c24'] * len(val)
            return [''] * len(val)
        
        # Affichage du tableau avec style
        st.dataframe(
            df_display[list(columns_display.keys())].rename(columns=columns_display),
            use_container_width=True,
            height=400,
            column_config={
                'Date Achat': st.column_config.DateColumn(
                    "Date Achat",
                    format="DD/MM/YYYY"
                ),
                'P&L (%)': st.column_config.TextColumn(
                    "P&L (%)",
                    help="Performance en pourcentage"
                ),
                'P&L (€)': st.column_config.TextColumn(
                    "P&L (€)",
                    help="Profit/Perte en euros"
                )
            }
        )
        
        # Résumé rapide
        total_positions = len(df_filtered)
        positions_rentables = len(df_filtered[df_filtered['pnl_pct'] > 0])
        taux_reussite = (positions_rentables / total_positions * 100) if total_positions > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Positions Affichées", total_positions)
        with col2:
            st.metric("✅ Positions Rentables", positions_rentables)
        with col3:
            st.metric("🎯 Taux de Réussite", f"{taux_reussite:.1f}%")


def create_sidebar_interface():
    """Interface sidebar avancée avec tous les formulaires"""
    
    with st.sidebar:
        # Logo et titre
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0;">🚀 OptiWealth</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">Gestion Professionnelle</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs pour organiser les actions
        tab1, tab2, tab3 = st.tabs(["➕ Ajouter", "📉 Vendre", "⚙️ Gestion"])
        
        with tab1:
            st.header("Nouvelle Position")
            
            with st.form("add_position_form", clear_on_submit=True):
                # Auto-complétion pour les symboles populaires
                popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "BTC-USD", "ETH-USD", 
                                 "MC.PA", "OR.PA", "SAN.PA", "BNP.PA", "AIR.PA"]
                
                col1, col2 = st.columns(2)
                with col1:
                    symbole = st.text_input(
                        "Symbole*",
                        placeholder="Ex: AAPL, BTC-USD",
                        help="Symbole Yahoo Finance"
                    ).upper()
                
                with col2:
                    if st.button("🔍 Vérifier"):
                        if symbole:
                            info = get_ticker_info(symbole)
                            if info.get('longName'):
                                st.success(f"✅ {info['longName']}")
                            else:
                                st.error("❌ Symbole non trouvé")
                
                # Suggestions de symboles populaires
                st.markdown("**Suggestions populaires:**")
                suggestion_cols = st.columns(5)
                for i, symbol in enumerate(popular_symbols[:10]):
                    with suggestion_cols[i % 5]:
                        if st.button(symbol, key=f"suggest_{symbol}"):
                            st.session_state.symbole_input = symbol
                
                nom = st.text_input(
                    "Nom de l'actif*",
                    placeholder="Ex: Apple Inc."
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    classe_actif = st.selectbox(
                        "Classe d'actif*",
                        options=list(ASSET_CLASSES.keys()),
                        format_func=lambda x: x
                    )
                
                with col2:
                    zone_geo = st.selectbox(
                        "Zone géographique",
                        options=GEOGRAPHICAL_ZONES
                    )
                
                secteur = st.text_input(
                    "Secteur",
                    placeholder="Ex: Technologie, Finance..."
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    montant = st.number_input(
                        "Montant/Quantité*",
                        min_value=0.01,
                        step=0.01,
                        format="%.4f",
                        help="Montant investi ou quantité achetée"
                    )
                
                with col2:
                    prix_achat = st.number_input(
                        "Prix d'achat*",
                        min_value=0.01,
                        step=0.01,
                        format="%.4f"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    frais = st.number_input(
                        "Frais",
                        min_value=0.0,
                        step=0.01,
                        value=0.0
                    )
                
                with col2:
                    date_achat = st.date_input(
                        "Date d'achat*",
                        value=datetime.now().date(),
                        max_value=datetime.now().date()
                    )
                
                # Calcul automatique
                if montant > 0 and prix_achat > 0:
                    valeur_totale = montant * prix_achat + frais
                    st.info(f"💰 Valeur totale: **{valeur_totale:,.2f} €**")
                
                submitted = st.form_submit_button("➕ Ajouter Position", use_container_width=True)
                
                if submitted:
                    if not all([symbole, nom, montant > 0, prix_achat > 0]):
                        st.error("❌ Veuillez remplir tous les champs obligatoires (*)")
                    else:
                        # Vérification du symbole
                        current_price = get_current_price(symbole)
                        if current_price is not None:
                            # Récupération des infos automatiques
                            info = get_ticker_info(symbole)
                            
                            new_position = pd.DataFrame({
                                'symbole': [symbole],
                                'nom': [nom],
                                'classe_actif': [ASSET_CLASSES[classe_actif]],
                                'montant': [montant],
                                'prix_achat': [prix_achat],
                                'date_achat': [pd.to_datetime(date_achat)],
                                'frais': [frais],
                                'zone_geo': [zone_geo],
                                'secteur': [secteur if secteur else info.get('sector', '')]
                            })
                            
                            df_portfolio = load_portfolio_data()
                            df_portfolio = pd.concat([df_portfolio, new_position], ignore_index=True)
                            save_portfolio_data(df_portfolio)
                            
                            st.success(f"✅ Position {symbole} ajoutée avec succès!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ Symbole {symbole} non trouvé sur Yahoo Finance")
        
        with tab2:
            st.header("Vendre Position")
            
            df_portfolio = load_portfolio_data()
            df_ouvertes = df_portfolio[df_portfolio['montant'] > 0]
            
            if df_ouvertes.empty:
                st.info("Aucune position à vendre")
            else:
                with st.form("sell_position_form"):
                    # Sélection par symbole
                    symbole_vente = st.selectbox(
                        "Symbole à vendre",
                        options=df_ouvertes['symbole'].unique()
                    )
                    
                    # Positions disponibles pour ce symbole
                    positions_symbole = df_ouvertes[df_ouvertes['symbole'] == symbole_vente]
                    
                    if len(positions_symbole) > 1:
                        position_idx = st.selectbox(
                            "Quelle position?",
                            options=positions_symbole.index,
                            format_func=lambda x: f"Achat {positions_symbole.loc[x, 'date_achat'].strftime('%d/%m/%Y')} - {positions_symbole.loc[x, 'montant']:.2f}"
                        )
                    else:
                        position_idx = positions_symbole.index[0]
                    
                    position_selectionnee = df_portfolio.loc[position_idx]
                    montant_max = position_selectionnee['montant']
                    
                    # Affichage des infos de la position
                    st.info(f"""
                    **Position sélectionnée:**
                    - Montant disponible: {montant_max:.4f}
                    - Prix d'achat: {position_selectionnee['prix_achat']:.2f} €
                    - Date d'achat: {position_selectionnee['date_achat'].strftime('%d/%m/%Y')}
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        montant_vente = st.number_input(
                            "Quantité à vendre",
                            min_value=0.01,
                            max_value=float(montant_max),
                            step=0.01,
                            value=float(montant_max)
                        )
                    
                    with col2:
                        # Prix de vente avec suggestion du prix actuel
                        prix_actuel = get_current_price(symbole_vente)
                        prix_vente = st.number_input(
                            "Prix de vente",
                            min_value=0.01,
                            step=0.01,
                            value=float(prix_actuel) if prix_actuel else position_selectionnee['prix_achat']
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        frais_vente = st.number_input(
                            "Frais de vente",
                            min_value=0.0,
                            step=0.01,
                            value=0.0
                        )
                    
                    with col2:
                        date_vente = st.date_input(
                            "Date de vente",
                            value=datetime.now().date(),
                            max_value=datetime.now().date()
                        )
                    
                    # Calcul du P&L prévisionnel
                    if montant_vente > 0 and prix_vente > 0:
                        pnl_previsionnel = montant_vente * (prix_vente - position_selectionnee['prix_achat']) - frais_vente
                        pnl_pct_prev = (pnl_previsionnel / (montant_vente * position_selectionnee['prix_achat']) * 100)
                        
                        color = "green" if pnl_previsionnel >= 0 else "red"
                        st.markdown(f"""
                        <div style="padding: 1rem; background: rgba(0,0,0,0.1); border-radius: 8px; text-align: center;">
                            <h4>📊 P&L Prévisionnel</h4>
                            <h2 style="color: {color};">{pnl_previsionnel:+.2f} € ({pnl_pct_prev:+.1f}%)</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    sell_submitted = st.form_submit_button("📉 Vendre Position", use_container_width=True)
                    
                    if sell_submitted:
                        if montant_vente <= 0 or montant_vente > montant_max:
                            st.error("❌ Quantité de vente invalide")
                        elif prix_vente <= 0:
                            st.error("❌ Prix de vente invalide")
                        else:
                            # Mise à jour de la position existante
                            df_portfolio.loc[position_idx, 'montant'] -= montant_vente
                            
                            # Enregistrement de la vente
                            vente_position = pd.DataFrame({
                                'symbole': [symbole_vente],
                                'nom': [position_selectionnee['nom']],
                                'classe_actif': [position_selectionnee['classe_actif']],
                                'montant': [-montant_vente],
                                'prix_achat': [prix_vente],  # Prix de vente stocké comme prix d'achat négatif
                                'date_achat': [pd.to_datetime(date_vente)],
                                'frais': [frais_vente],
                                'zone_geo': [position_selectionnee['zone_geo']],
                                'secteur': [position_selectionnee['secteur']]
                            })
                            
                            df_portfolio = pd.concat([df_portfolio, vente_position], ignore_index=True)
                            
                            # Supprimer la ligne si montant = 0
                            if df_portfolio.loc[position_idx, 'montant'] == 0:
                                df_portfolio = df_portfolio.drop(position_idx).reset_index(drop=True)
                            
                            save_portfolio_data(df_portfolio)
                            
                            st.success(f"✅ Vente réalisée: {pnl_previsionnel:+.2f} € ({pnl_pct_prev:+.1f}%)")
                            st.balloons()
                            st.rerun()
        
        with tab3:
            st.header("Gestion du Portfolio")
            
            df_portfolio = load_portfolio_data()
            
            # Export des données
            if not df_portfolio.empty:
                st.subheader("📤 Export")
                
                # Export CSV
                csv = df_portfolio.to_csv(index=False)
                st.download_button(
                    "💾 Télécharger CSV",
                    csv,
                    "portfolio_export.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Statistiques du fichier
                file_stats = {
                    "Taille du fichier": f"{len(csv)} caractères",
                    "Nombre de positions": len(df_portfolio),
                    "Dernière modification": datetime.fromtimestamp(os.path.getmtime(CSV_FILE)).strftime('%d/%m/%Y %H:%M') if os.path.exists(CSV_FILE) else "N/A"
                }
                
                for stat, value in file_stats.items():
                    st.text(f"{stat}: {value}")
            
            st.subheader("🗑️ Suppression")
            
            if not df_portfolio.empty:
                position_to_delete = st.selectbox(
                    "Position à supprimer",
                    options=df_portfolio.index,
                    format_func=lambda x: f"{df_portfolio.loc[x, 'symbole']} - {df_portfolio.loc[x, 'nom']} ({df_portfolio.loc[x, 'date_achat'].strftime('%d/%m/%Y')})"
                )
                
                if st.button("🗑️ Supprimer Position", use_container_width=True):
                    df_portfolio = df_portfolio.drop(position_to_delete).reset_index(drop=True)
                    save_portfolio_data(df_portfolio)
                    st.success("✅ Position supprimée!")
                    st.rerun()
            else:
                st.info("Aucune position à supprimer")
            
            # Nettoyage des données
            st.subheader("🧹 Maintenance")
            
            if st.button("🔄 Nettoyer les doublons", use_container_width=True):
                if not df_portfolio.empty:
                    df_clean = df_portfolio.drop_duplicates(subset=['symbole', 'date_achat', 'prix_achat'])
                    removed = len(df_portfolio) - len(df_clean)
                    if removed > 0:
                        save_portfolio_data(df_clean)
                        st.success(f"✅ {removed} doublons supprimés!")
                        st.rerun()
                    else:
                        st.info("Aucun doublon détecté")
            
            # Sauvegarde de sécurité
            if st.button("💾 Sauvegarde de sécurité", use_container_width=True):
                backup_name = f"portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_portfolio.to_csv(backup_name, index=False)
                st.success(f"✅ Sauvegarde créée: {backup_name}")


# Interface principale
def main():
    """Fonction principale du dashboard"""
    
    # Chargement et traitement des données
    df_portfolio = load_portfolio_data()
    
    # Interface sidebar
    create_sidebar_interface()
    
    # Contenu principal
    if df_portfolio.empty:
        # Page d'accueil pour nouveaux utilisateurs
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🚀 Bienvenue sur OptiWealth Pro</h1>
            <p class="header-subtitle">Votre tableau de bord professionnel pour gérer vos investissements</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 15px; margin-bottom: 1rem;">
                <h3>📊 Suivi en Temps Réel</h3>
                <p>Prix actualisés via Yahoo Finance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 15px; margin-bottom: 1rem;">
                <h3>💰 Calculs Automatiques</h3>
                <p>P&L, performances, métriques</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 15px; margin-bottom: 1rem;">
                <h3>📈 Visualisations</h3>
                <p>Graphiques interactifs avancés</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("👈 Commencez par ajouter votre première position via la barre latérale!")
        
        # Guide de démarrage
        with st.expander("📚 Guide de démarrage rapide"):
            st.markdown("""
            ### 🚀 Comment commencer ?
            
            1. **➕ Ajouter une position** : Utilisez l'onglet "Ajouter" dans la barre latérale
            2. **📊 Suivre les performances** : Le dashboard se met à jour automatiquement
            3. **📉 Gérer les ventes** : Vendez partiellement ou totalement vos positions
            4. **📈 Analyser** : Explorez les graphiques et métriques avancées
            
            ### 💡 Symboles supportés
            - **Actions US** : AAPL, MSFT, GOOGL, AMZN, TSLA...
            - **Actions EU** : MC.PA, ASML.AS, SAP.DE...
            - **Crypto** : BTC-USD, ETH-USD, ADA-USD...
            - **ETF** : SPY, QQQ, VTI, IWDA.AS...
            
            ### 🔧 Fonctionnalités avancées
            - Calculs automatiques de P&L
            - Métriques de risque (Sharpe, volatilité)
            - Analyse de diversification
            - Recommandations intelligentes
            - Export des données
            """)
    
    else:
        # Dashboard principal avec données
        df_portfolio, df_ouvertes, df_ventes = create_advanced_metrics_display(df_portfolio)
        
        # Espacement
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualisations avancées
        create_advanced_visualizations(df_portfolio, df_ouvertes)
        
        # Espacement
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tableau détaillé
        create_portfolio_table(df_portfolio)
    
    # Footer avec informations
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("💡 **Données** : Sauvegarde automatique en CSV")
    with col2:
        st.markdown("🔄 **Prix** : Mise à jour via Yahoo Finance")
    with col3:
        if st.button("🔄 Actualiser", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# Lancement de l'application
if __name__ == "__main__":
    main()