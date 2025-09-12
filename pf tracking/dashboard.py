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