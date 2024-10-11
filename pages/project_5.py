import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Paramètres fixes pour la génération de données
START_DATE = datetime(2020, 1, 1)
NUM_DAYS = 1000
BASE_EXPORT = 1000
TREND = 0.1
SEASONALITY = 0.2
NOISE = 0.05

def generate_export_data(date):
    t = (date - START_DATE).days
    trend_component = TREND * t
    seasonal_component = SEASONALITY * np.sin(2 * np.pi * t / 365)
    noise_component = NOISE * np.random.randn()
    export = BASE_EXPORT + trend_component + seasonal_component * BASE_EXPORT + noise_component * BASE_EXPORT
    return max(0, int(export))

def generate_sample_data():
    dates = [START_DATE + timedelta(days=i) for i in range(NUM_DAYS)]
    exports = [generate_export_data(date) for date in dates]
    return pd.DataFrame({'ds': dates, 'y': exports})

def main():
    st.title("Prédiction de Séries Temporelles avec Prophet")
    
    with st.expander("À propos de Prophet"):
        st.markdown("""
        ### Prophet : Des prévisions explicables à grande échelle

        Prophet est un outil open-source développé par Facebook Research, conçu pour la prévision de séries temporelles. NeuralProphet comble le fossé entre les modèles de séries temporelles traditionnels et les méthodes d'apprentissage profond. Il est basé sur PyTorch.

        #### Caractéristiques principales
        - **Décomposition additive** : Combine tendance, saisonnalité et jours fériés
        - **Robuste aux données manquantes** : Gère bien les irrégularités dans les données
        - **Gestion des changements de tendance** : Détecte les points de changement
        - **Saisonnalités multiples** : Gère les motifs quotidiens, hebdomadaires et annuels
        """)

    # Options pour les données
    data_option = st.radio(
        "Choisissez une source de données",
        ("Utiliser des données générées", "Télécharger vos propres données")
    )

    df = None
    if data_option == "Télécharger vos propres données":
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Données chargées avec succès!")
                
                # Sélection des colonnes
                date_col = st.selectbox("Sélectionnez la colonne de date", df.columns)
                value_col = st.selectbox("Sélectionnez la colonne de valeurs", [col for col in df.columns if col != date_col])
                
                # Préparation des données
                df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
                df['ds'] = pd.to_datetime(df['ds'])
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {str(e)}")
    else:
        st.write("Utilisation de données générées")
        df = generate_sample_data()
        st.success("Données générées avec succès!")

    if df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        
        fig = px.line(df, x='ds', y='y', title="Série temporelle")
        st.plotly_chart(fig)

        # Configuration du modèle
        st.subheader("Configuration du modèle")
        n_forecasts = st.slider("Nombre de périodes à prédire", 7, 90, 30)
        yearly_seasonality = st.checkbox("Saisonnalité annuelle", value=True)
        weekly_seasonality = st.checkbox("Saisonnalité hebdomadaire", value=True)
        daily_seasonality = st.checkbox("Saisonnalité journalière", value=False)

        if st.button("Entraîner le modèle et prédire"):
            with st.spinner("Entraînement du modèle en cours..."):
                # Création et configuration du modèle
                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                )
                
                # Entraînement
                model.fit(df)
                
                # Prédiction
                future = model.make_future_dataframe(periods=n_forecasts)
                forecast = model.predict(future)
                
                st.success("Modèle entraîné avec succès!")
                
                # Affichage des résultats
                st.subheader("Résultats de la prédiction")
                
                # Graphique des prédictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Données réelles'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prédictions'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Limite supérieure', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Limite inférieure', line=dict(dash='dash')))
                fig.update_layout(title='Prédictions vs Données réelles', xaxis_title='Date', yaxis_title='Valeur')
                st.plotly_chart(fig)
                
                # Composantes du modèle
                st.subheader("Composantes du modèle")
                fig_comp = model.plot_components(forecast)
                st.pyplot(fig_comp)
                
                # Métriques de performance
                st.subheader("Métriques de performance")
                mae = np.mean(np.abs(forecast['yhat'][:len(df)] - df['y']))
                rmse = np.sqrt(np.mean((forecast['yhat'][:len(df)] - df['y'])**2))
                
                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                
                # Option de téléchargement des prédictions
                csv = forecast.to_csv(index=False)
                st.download_button(
                    "Télécharger les prédictions (CSV)",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )

if __name__ == "__main__":
    main()