import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from PIL import Image

# Chargement du logo avec PIL (chemin relatif + gestion d'erreur)
try:
    logo = Image.open("logo_final.jpg")
except Exception:
    logo = None

# Configuration de la page
if logo is not None:
    st.set_page_config(
        page_title="OWS DashBoard",
        page_icon=logo,
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="OWS DashBoard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

PORTFOLIO_DIR = "portfolios"
# Créer le dossier s'il n'existe pas
if not os.path.exists(PORTFOLIO_DIR):
    os.makedirs(PORTFOLIO_DIR)

# Classes d'actifs disponibles
ASSET_CLASSES = [
    "Actions",
    "ETF",
    "Crypto",
    "Obligations",
    "Matières Premières",
    "Autre"
]

def load_portfolio_data(portfolio_name):
    """Charge les données du portfolio depuis le fichier CSV spécifique"""
    csv_file = f"{PORTFOLIO_DIR}/{portfolio_name}.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            df['date_achat'] = pd.to_datetime(df['date_achat'])
            # Vérifier si les colonnes 'montant' et 'frais' existent, sinon les ajouter avec 0.0
            updated = False
            if 'montant' not in df.columns:
                df['montant'] = 0.0
                updated = True
            if 'frais' not in df.columns:
                df['frais'] = 0.0
                updated = True
            if updated:
                save_portfolio_data(df, portfolio_name)
            return df
        except:
            return pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais'])
    else:
        return pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais'])

def save_portfolio_data(df, portfolio_name):
    """Sauvegarde les données du portfolio dans le fichier CSV spécifique"""
    csv_file = f"{PORTFOLIO_DIR}/{portfolio_name}.csv"
    df.to_csv(csv_file, index=False)

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

st.title("OWS DashBoard")

# --- Gestion multi-portefeuilles ---
with st.sidebar:
    if logo is not None:
        st.sidebar.image(logo, width=150)
    st.header("🗂️ Sélection du portefeuille")
    # Liste des portefeuilles existants (fichiers .csv dans le dossier)
    existing_portfolios = [f[:-4] for f in os.listdir(PORTFOLIO_DIR) if f.endswith('.csv')]
    if not existing_portfolios:
        existing_portfolios = ["MonPortefeuille"]
        # Créer un portefeuille vide par défaut
        pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais']).to_csv(
            f"{PORTFOLIO_DIR}/MonPortefeuille.csv", index=False
        )
    selected_portfolio = st.selectbox("Portefeuille courant", existing_portfolios, key="portfolio_selectbox")
    st.markdown("---")
    st.subheader("Créer un nouveau portefeuille")
    new_portfolio_name = st.text_input("Nom du nouveau portefeuille", key="new_portfolio_name")
    if st.button("Créer le portefeuille"):
        if new_portfolio_name.strip() == "":
            st.warning("Veuillez saisir un nom de portefeuille.")
        elif new_portfolio_name in existing_portfolios:
            st.warning("Ce portefeuille existe déjà.")
        else:
            pd.DataFrame(columns=['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'date_achat', 'frais']).to_csv(
                f"{PORTFOLIO_DIR}/{new_portfolio_name}.csv", index=False
            )
            st.success(f"Portefeuille '{new_portfolio_name}' créé !")
            st.rerun()

# Chargement des données du portefeuille sélectionné
df_portfolio = load_portfolio_data(selected_portfolio)

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
                save_portfolio_data(df_portfolio, selected_portfolio)
                st.success(f"Position {symbole} ajoutée avec succès!")
                st.rerun()
            else:
                st.error(f"Symbole {symbole} non trouvé sur Yahoo Finance")

    # Gestion du P&L réalisé cumulé (persisté dans session_state)
    if "pnl_realise_total" not in st.session_state:
        st.session_state["pnl_realise_total"] = 0.0
    if not df_portfolio.empty:
        st.header("📉 Vendre une position")
        with st.form("sell_position_form"):
            sell_symbole = st.selectbox(
                "Symbole",
                options=df_portfolio['symbole'].unique(),
                key="sell_symbole_select"
            )
            filtered_positions = df_portfolio[df_portfolio['symbole'] == sell_symbole]
            position_indices = filtered_positions.index.tolist()
            position_display = [f"{row['nom']} - Montant: {row['montant']}" for _, row in filtered_positions.iterrows()]
            selected_position_idx = st.selectbox(
                "Sélectionner la ligne à vendre",
                options=position_indices,
                format_func=lambda x: position_display[position_indices.index(x)],
                key="sell_position_line_select"
            )
            max_montant = df_portfolio.loc[selected_position_idx, 'montant']
            montant_vente = st.number_input(
                "Montant à vendre",
                min_value=0.01,
                max_value=float(max_montant),
                step=0.01,
                key="montant_vente_input"
            )
            prix_vente = st.number_input(
                "Prix de vente",
                min_value=0.0,
                step=0.01,
                key="prix_vente_input"
            )
            frais_vente = st.number_input(
                "Frais de vente",
                min_value=0.0,
                step=0.01,
                key="frais_vente_input"
            )
            date_vente = st.date_input(
                "Date de vente",
                value=datetime.now().date(),
                key="date_vente_input"
            )
            sell_submitted = st.form_submit_button("Vendre la position")
            if sell_submitted:
                if montant_vente <= 0:
                    st.error("Le montant à vendre doit être supérieur à 0.")
                elif montant_vente > max_montant:
                    st.error("Le montant à vendre ne peut pas dépasser le montant détenu.")
                else:
                    # Si prix_vente = 0, utiliser le prix actuel
                    if prix_vente == 0.0:
                        prix_vente_effectif = get_current_price(df_portfolio.loc[selected_position_idx, 'symbole'])
                        if prix_vente_effectif is None or prix_vente_effectif == 0.0:
                            st.error("Impossible de récupérer le prix actuel pour effectuer la vente.")
                            st.stop()
                    else:
                        prix_vente_effectif = prix_vente

                    prix_achat_orig = df_portfolio.loc[selected_position_idx, 'prix_achat']

                    # Calcul du PnL réalisé sur la vente
                    pnl_vente = montant_vente * (prix_vente_effectif - prix_achat_orig) - frais_vente

                    # Mise à jour cumulative du P&L réalisé
                    st.session_state["pnl_realise_total"] += pnl_vente

                    # Si vente totale, suppression de la ligne
                    if abs(montant_vente - max_montant) < 1e-8:
                        df_portfolio = df_portfolio.drop(selected_position_idx).reset_index(drop=True)
                    else:
                        # Vente partielle, décrémenter le montant
                        nouveau_montant = df_portfolio.loc[selected_position_idx, 'montant'] - montant_vente
                        df_portfolio.loc[selected_position_idx, 'montant'] = nouveau_montant

                    # NE PLUS ajouter de ligne de vente avec montant négatif
                    save_portfolio_data(df_portfolio, selected_portfolio)
                    st.success(f"Vente de {montant_vente} {sell_symbole} effectuée avec succès! P&L réalisé: {pnl_vente:.2f} €")
                    st.rerun()

    if not df_portfolio.empty:
        st.header("🗑️ Supprimer une position")
        position_to_delete = st.selectbox(
            "Sélectionner la position à supprimer",
            options=df_portfolio.index,
            format_func=lambda x: f"{df_portfolio.loc[x, 'symbole']} - {df_portfolio.loc[x, 'nom']}"
        )
        if st.button("Supprimer la position"):
            df_portfolio = df_portfolio.drop(position_to_delete).reset_index(drop=True)
            save_portfolio_data(df_portfolio, selected_portfolio)
            st.success("Position supprimée!")
            st.rerun()

# Contenu principal
if df_portfolio.empty:
    st.info("Aucune position dans votre portefeuille. Ajoutez votre première position via la barre latérale.")
else:
    # Récupération des prix actuels
    with st.spinner("Récupération des prix actuels..."):
        current_prices = []
        for symbol in df_portfolio['symbole']:
            price = get_current_price(symbol)
            current_prices.append(price if price is not None else 0)
        df_portfolio['prix_actuel'] = current_prices
    
    # Calcul des P&L prenant en compte les ventes partielles et les frais
    # Calcul des P&L avec 'montant' comme somme investie et non quantité
        # -------------------------
# CALCUL DES METRICS PORTFOLIO
    # -------------------------

    # Séparer positions ouvertes
    df_ouvertes = df_portfolio[df_portfolio['montant'] > 0].copy()

    # Pour les positions ouvertes : P&L flottant
    df_ouvertes['valeur_achat'] = df_ouvertes['montant'] + df_ouvertes['frais']
    df_ouvertes['valeur_actuelle'] = df_ouvertes['montant'] * (df_ouvertes['prix_actuel'] / df_ouvertes['prix_achat'])
    df_ouvertes['pnl_absolu'] = df_ouvertes['valeur_actuelle'] - df_ouvertes['valeur_achat']
    df_ouvertes['pnl_pct'] = np.where(
        df_ouvertes['valeur_achat'] != 0,
        (df_ouvertes['pnl_absolu'] / df_ouvertes['valeur_achat'] * 100).round(2),
        0.0
    )

    # Metrics globales
    capital_investi = df_ouvertes['valeur_achat'].sum()
    valeur_actuelle = df_ouvertes['valeur_actuelle'].sum()
    pnl_flottant = df_ouvertes['pnl_absolu'].sum()
    # Utiliser le P&L réalisé cumulé en session_state
    pnl_realise = st.session_state.get("pnl_realise_total", 0.0)
    nb_positions = len(df_ouvertes)

    # -------------------------
    # AFFICHAGE HEADER
    # -------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💰 Capital Investi", f"{capital_investi:,.2f} €")

    with col2:
        st.metric("📈 Valeur Actuelle", f"{valeur_actuelle:,.2f} €")

    with col3:
        pnl_color = "normal" if pnl_flottant >= 0 else "inverse"
        st.metric("📊 P&L Flottant", f"{pnl_flottant:,.2f} €", delta_color=pnl_color)

    with col4:
        pnl_color_real = "normal" if pnl_realise >= 0 else "inverse"
        st.metric("💸 P&L Réalisé", f"{pnl_realise:,.2f} €", delta_color=pnl_color_real)

    with col5:
        st.metric("🎯 Nb Positions Ouvertes", nb_positions)

    # -------------------------
    # PREPARATION DU TABLEAU COMPLET POUR AFFICHAGE
    # -------------------------
    df_display = df_ouvertes.copy()
    df_display['prix_achat'] = df_display['prix_achat'].map('{:.2f} €'.format)
    df_display['prix_actuel'] = df_display['prix_actuel'].map('{:.2f} €'.format)
    df_display['valeur_achat'] = df_display['valeur_achat'].map('{:,.2f} €'.format)
    df_display['valeur_actuelle'] = df_display['valeur_actuelle'].map('{:,.2f} €'.format)
    df_display['pnl_absolu'] = df_display['pnl_absolu'].map('{:,.2f} €'.format)
    df_display['pnl_pct'] = df_display['pnl_pct'].map('{:.2f}%'.format)
    df_display['frais'] = df_display['frais'].map('{:.2f} €'.format)

    columns_to_show = ['symbole', 'nom', 'classe_actif', 'montant', 'prix_achat', 'frais',
                    'prix_actuel', 'valeur_actuelle', 'pnl_absolu', 'pnl_pct']

    st.header("📋 Portfolio Détaillé")
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
        # Répartition par classe d'actif (ne prendre en compte que les positions ouvertes)
        df_allocation = df_ouvertes.groupby('classe_actif')['valeur_actuelle'].sum().reset_index()
        fig_pie = px.pie(df_allocation, values='valeur_actuelle', names='classe_actif',
                        title="Répartition par Classe d'Actif")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # P&L par position (uniquement sur positions ouvertes)
        if 'pnl_absolu' in df_display.columns:
            fig_bar = px.bar(df_display, x='symbole', y='pnl_absolu',
                            color='pnl_absolu', color_continuous_scale='RdYlGn',
                            title="P&L par Position")
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Impossible d'afficher le P&L par position (données manquantes).")
    
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
st.markdown(f"💡 **Astuce**: Les données sont automatiquement sauvegardées dans le dossier `{PORTFOLIO_DIR}`")
st.markdown("🔄 **Mise à jour**: Les prix sont récupérés depuis Yahoo Finance à chaque actualisation")