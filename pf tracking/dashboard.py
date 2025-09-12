import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration de la page
st.set_page_config(
    page_title="Dashboard d'Investissement",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fichier CSV pour sauvegarder les données
CSV_FILE = "portfolio_data.csv"

# Classes d'actifs disponibles
ASSET_CLASSES = [
    "Actions",
    "ETF",
    "Crypto",
    "Obligations",
    "Matières Premières",
    "Autre"
]

def load_portfolio_data():
    """Charge les données du portfolio depuis le fichier CSV"""
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df['date_achat'] = pd.to_datetime(df['date_achat'])
            return df
        except:
            return pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais'])
    else:
        return pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais'])

def save_portfolio_data(df):
    """Sauvegarde les données du portfolio dans le fichier CSV"""
    df.to_csv(CSV_FILE, index=False)

def get_current_price(symbol):
    """Récupère le prix actuel depuis Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            return None
    except:
        return None

def get_historical_data(symbol, period="1y"):
    """Récupère les données historiques"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except:
        return None

def calculate_performance_metrics(df_portfolio):
    """Calcule les métriques de performance"""
    if df_portfolio.empty:
        return {}
    
    total_invested = (df_portfolio['montant'] * df_portfolio['prix_achat'] + df_portfolio['frais']).sum()
    total_current_value = (df_portfolio['montant'] * df_portfolio['prix_actuel']).sum()
    total_pnl = total_current_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    return {
        'total_invested': total_invested,
        'total_current_value': total_current_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'nb_positions': len(df_portfolio)
    }

# Interface principale
st.title("💰 Dashboard d'Investissement")

# Chargement des données
df_portfolio = load_portfolio_data()

# Sidebar pour ajouter des positions
with st.sidebar:
    st.header("➕ Ajouter une position")
    
    with st.form("add_position_form"):
        symbole = st.text_input("Symbole (ex: AAPL, BTC-USD)", key="symbole_input")
        nom = st.text_input("Nom de l'actif", key="nom_input")
        classe_actif = st.selectbox("Classe d'actif", ASSET_CLASSES, key="classe_input")
        montant = st.number_input("Montant", min_value=0.0, step=0.01, key="montant_input")
        prix_achat = st.number_input("Prix d'achat", min_value=0.0, step=0.01, key="prix_input")
        frais = st.number_input("Frais", min_value=0.0, step=0.01, key="frais_input")
        date_achat = st.date_input("Date d'achat", value=datetime.now().date(), key="date_input")
        
        submitted = st.form_submit_button("Ajouter la position")
        
        if submitted and symbole and nom and montant > 0 and prix_achat > 0:
            # Vérifier si le symbole existe
            current_price = get_current_price(symbole)
            if current_price is not None:
                new_position = pd.DataFrame({
                    'symbole': [symbole.upper()],
                    'nom': [nom],
                    'classe_actif': [classe_actif],
                    'montant': [montant],
                    'prix_achat': [prix_achat],
                    'date_achat': [pd.to_datetime(date_achat)],
                    'frais': [frais]
                })
                
                df_portfolio = pd.concat([df_portfolio, new_position], ignore_index=True)
                save_portfolio_data(df_portfolio)
                st.success(f"Position {symbole} ajoutée avec succès!")
                st.rerun()
            else:
                st.error(f"Symbole {symbole} non trouvé sur Yahoo Finance")

    # Bouton pour supprimer une position
    if not df_portfolio.empty:
        st.header("🗑️ Supprimer une position")
        position_to_delete = st.selectbox(
            "Sélectionner la position à supprimer",
            options=df_portfolio.index,
            format_func=lambda x: f"{df_portfolio.loc[x, 'symbole']} - {df_portfolio.loc[x, 'nom']}"
        )
        
        if st.button("Supprimer la position"):
            df_portfolio = df_portfolio.drop(position_to_delete).reset_index(drop=True)
            save_portfolio_data(df_portfolio)
            st.success("Position supprimée!")
            st.rerun()

# Contenu principal
if df_portfolio.empty:
    st.info("Aucune position dans votre portfolio. Ajoutez votre première position via la barre latérale.")
else:
    # Récupération des prix actuels
    with st.spinner("Récupération des prix actuels..."):
        current_prices = []
        for symbol in df_portfolio['symbole']:
            price = get_current_price(symbol)
            current_prices.append(price if price is not None else 0)
        
        df_portfolio['prix_actuel'] = current_prices
    
    # Calcul des P&L
    df_portfolio['valeur_achat'] = df_portfolio['montant'] * df_portfolio['prix_achat'] + df_portfolio['frais']
    df_portfolio['valeur_actuelle'] = df_portfolio['montant'] * df_portfolio['prix_actuel']
    df_portfolio['pnl_absolu'] = df_portfolio['valeur_actuelle'] - df_portfolio['valeur_achat']
    df_portfolio['pnl_pct'] = (df_portfolio['pnl_absolu'] / df_portfolio['valeur_achat'] * 100).round(2)
    
    # Métriques de performance
    metrics = calculate_performance_metrics(df_portfolio)
    
    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Valeur Investie", f"{metrics['total_invested']:,.2f} €")
    
    with col2:
        st.metric("📈 Valeur Actuelle", f"{metrics['total_current_value']:,.2f} €")
    
    with col3:
        pnl_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
        st.metric("📊 P&L Total", f"{metrics['total_pnl']:,.2f} €", 
                 delta=f"{metrics['total_pnl_pct']:.2f}%", delta_color=pnl_color)
    
    with col4:
        st.metric("🎯 Nb Positions", metrics['nb_positions'])
    
    # Tableau du portfolio
    st.header("📋 Portfolio Détaillé")
    
    # Formatage pour l'affichage
    df_display = df_portfolio.copy()
    df_display['prix_achat'] = df_display['prix_achat'].map('{:.2f} €'.format)
    df_display['prix_actuel'] = df_display['prix_actuel'].map('{:.2f} €'.format)
    df_display['valeur_achat'] = df_display['valeur_achat'].map('{:,.2f} €'.format)
    df_display['valeur_actuelle'] = df_display['valeur_actuelle'].map('{:,.2f} €'.format)
    df_display['pnl_absolu'] = df_display['pnl_absolu'].map('{:,.2f} €'.format)
    df_display['pnl_pct'] = df_display['pnl_pct'].map('{:.2f}%'.format)
    df_display['frais'] = df_display['frais'].map('{:.2f} €'.format)
    
    # Colonnes à afficher
    columns_to_show = ['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'frais',
                      'prix_actuel', 'valeur_actuelle', 'pnl_absolu', 'pnl_pct']
    
    st.dataframe(
        df_display[columns_to_show],
        column_config={
            'symbole': 'Symbole',
            'nom': 'Nom',
            'classe_actif': 'Classe',
            'montant': 'Montant',
            'prix_achat': 'Prix Achat',
            'frais': 'Frais',
            'prix_actuel': 'Prix Actuel',
            'valeur_actuelle': 'Valeur Actuelle',
            'pnl_absolu': 'P&L (€)',
            'pnl_pct': 'P&L (%)'
        },
        use_container_width=True
    )
    
    # Graphiques
    st.header("📊 Visualisations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition par classe d'actif
        df_allocation = df_portfolio.groupby('classe_actif')['valeur_actuelle'].sum().reset_index()
        fig_pie = px.pie(df_allocation, values='valeur_actuelle', names='classe_actif',
                        title="Répartition par Classe d'Actif")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # P&L par position
        fig_bar = px.bar(df_portfolio, x='symbole', y='pnl_absolu',
                        color='pnl_absolu', color_continuous_scale='RdYlGn',
                        title="P&L par Position")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Graphique d'évolution historique pour une position sélectionnée
    st.header("📈 Évolution Historique")
    
    selected_symbol = st.selectbox("Sélectionner une position", df_portfolio['symbole'].tolist())
    period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    if selected_symbol:
        historical_data = get_historical_data(selected_symbol, period)
        if historical_data is not None and not historical_data.empty:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name=f'{selected_symbol} Prix',
                line=dict(color='blue', width=2)
            ))
            
            # Ligne du prix d'achat
            purchase_price = df_portfolio[df_portfolio['symbole'] == selected_symbol]['prix_achat'].iloc[0]
            fig_line.add_hline(y=purchase_price, line_dash="dash", line_color="red",
                             annotation_text=f"Prix d'achat: {purchase_price:.2f}€")
            
            fig_line.update_layout(
                title=f"Évolution de {selected_symbol} - {period}",
                xaxis_title="Date",
                yaxis_title="Prix (€)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.error(f"Impossible de récupérer les données historiques pour {selected_symbol}")

# Bouton de refresh manuel
if st.button("🔄 Actualiser les Prix", type="primary"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 **Astuce**: Les données sont automatiquement sauvegardées dans le fichier `portfolio_data.csv`")
st.markdown("🔄 **Mise à jour**: Les prix sont récupérés depuis Yahoo Finance à chaque actualisation")