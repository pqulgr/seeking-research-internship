import streamlit as st
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    st.title("Prédiction de Séries Temporelles avec NeuralProphet")
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
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        values = np.random.randn(len(dates)).cumsum() + 100 + np.sin(np.arange(len(dates))/365*2*np.pi)*10
        df = pd.DataFrame({'ds': dates, 'y': values})
        st.success("Données générées avec succès!")

    if df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Données'))
        fig.update_layout(title='Série temporelle', xaxis_title='Date', yaxis_title='Valeur')
        st.plotly_chart(fig)

        # Configuration du modèle
        st.subheader("Configuration de la prédiction")
        n_forecasts = st.slider("Nombre de jours à prédire", 7, 365, 30)

        if st.button("Faire les prédictions"):
            with st.spinner("Calcul des prédictions en cours..."):
                # Création et entraînement du modèle
                model = NeuralProphet()
                metrics = model.fit(df, freq="D")
                
                # Prédiction
                future = model.make_future_dataframe(df, periods=n_forecasts)
                forecast = model.predict(future)
                
                st.success("Prédictions calculées avec succès!")
                
                # Affichage des résultats
                st.subheader("Résultats de la prédiction")
                
                # 1. Graphique des prédictions
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Données réelles'))
                fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name='Prédictions'))
                fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1_upper'], name='Limite supérieure', line=dict(dash='dash')))
                fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1_lower'], name='Limite inférieure', line=dict(dash='dash')))
                fig1.update_layout(title='Prédictions vs Données réelles', xaxis_title='Date', yaxis_title='Valeur')
                st.plotly_chart(fig1)
                
                # 2. Composantes du modèle
                components = model.get_latest_forecast_components(future)
                fig2 = make_subplots(rows=3, cols=1, subplot_titles=('Tendance', 'Saisonnalité annuelle', 'Saisonnalité hebdomadaire'))
                fig2.add_trace(go.Scatter(x=components['ds'], y=components['trend'], name='Tendance'), row=1, col=1)
                fig2.add_trace(go.Scatter(x=components['ds'], y=components['yearly'], name='Saisonnalité annuelle'), row=2, col=1)
                fig2.add_trace(go.Scatter(x=components['ds'], y=components['weekly'], name='Saisonnalité hebdomadaire'), row=3, col=1)
                fig2.update_layout(height=900, title_text="Composantes du modèle")
                st.plotly_chart(fig2)
                
                # 3. Erreurs de prédiction
                errors = df['y'].values - forecast['yhat1'][:len(df)].values
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df['ds'], y=errors, mode='markers', name='Erreurs'))
                fig3.update_layout(title='Erreurs de prédiction', xaxis_title='Date', yaxis_title='Erreur')
                st.plotly_chart(fig3)
                
                # Métriques de performance
                st.subheader("Précision du modèle")
                mae = np.mean(np.abs(errors))
                rmse = np.sqrt(np.mean(errors**2))
                
                col1, col2 = st.columns(2)
                col1.metric("Erreur moyenne absolue (MAE)", f"{mae:.2f}")
                col2.metric("Racine de l'erreur quadratique moyenne (RMSE)", f"{rmse:.2f}")
                
                # Option de téléchargement des prédictions
                csv = forecast[['ds', 'yhat1', 'yhat1_lower', 'yhat1_upper']].to_csv(index=False)
                st.download_button(
                    "Télécharger les prédictions (CSV)",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )

if __name__ == "__main__":
    main()