import streamlit as st
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Paramètres pour la génération de données
start_date = datetime(2024, 10, 10)

def generate_data(num_days):
    """Génère des données d'exportation avec tendance, saisonnalité et bruit"""
    base_export = 1000
    trend = 0.1
    seasonality = 0.2
    noise = 0.05
    
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    exports = []
    for date in dates:
        t = (date - start_date).days
        trend_component = trend * t
        seasonal_component = seasonality * np.sin(2 * np.pi * t / 365)
        noise_component = noise * np.random.randn()
        export = base_export + trend_component + seasonal_component * base_export + noise_component * base_export
        exports.append(max(0, int(export)))  # Assurer que les exportations ne sont jamais négatives
    
    return pd.DataFrame({'ds': dates, 'y': exports})

def plot_timeseries(df):
    """Affiche un graphique de la série temporelle"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Données'))
    fig.update_layout(title='Série temporelle', xaxis_title='Date', yaxis_title='Valeur')
    st.plotly_chart(fig)

def make_predictions(df, n_forecasts):
    """Fait des prédictions avec NeuralProphet"""
    model = NeuralProphet()
    model.fit(df, freq='D')
    
    future = model.make_future_dataframe(df, periods=n_forecasts)
    forecast = model.predict(future)
    
    return forecast

def plot_forecast(df, forecast):
    """Affiche les prédictions par rapport aux données réelles"""
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Données réelles'))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name='Prédictions'))
    fig1.update_layout(title='Prédictions vs Données réelles', xaxis_title='Date', yaxis_title='Valeur')
    st.plotly_chart(fig1)

def main():
    st.title("Prédiction de Séries Temporelles avec NeuralProphet")
    
    if 'export_data' not in st.session_state:
        st.session_state['export_data'] = None
    
    # Choix de la source des données
    data_option = st.radio("Choisissez une source de données", ("Utiliser des données générées", "Télécharger vos propres données"))
    
    df = None  # Initialise df pour éviter les erreurs avant le chargement des données

    # Chargement des données (CSV ou XLSX)
    if data_option == "Télécharger vos propres données":
        uploaded_file = st.file_uploader("Choisissez un fichier CSV ou XLSX", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                # Charger le fichier selon son extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                
                st.success("Données chargées avec succès!")
                
                # Sélection des colonnes
                date_col = st.selectbox("Sélectionnez la colonne de date", df.columns)
                value_col = st.selectbox("Sélectionnez la colonne de valeurs", [col for col in df.columns if col != date_col])
                
                df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
                df['ds'] = pd.to_datetime(df['ds'])
                st.session_state['export_data'] = df
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {str(e)}")
    
    else:
        # Génération de données synthétiques
        num_days = st.slider('Nombre de jours de données à générer', 1, 1000, 100)
        if st.session_state['export_data'] is None or len(st.session_state['export_data']) != num_days:
            st.session_state['export_data'] = generate_data(num_days)
        df = st.session_state['export_data']
    
    # Affichage des données
    if df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        plot_timeseries(df)
    
        # Prédictions
        n_forecasts = st.slider("Nombre de jours à prédire", 7, 365, 30)
        if st.button("Faire les prédictions"):
            with st.spinner("Calcul des prédictions en cours..."):
                forecast = make_predictions(df, n_forecasts)
                st.success("Prédictions calculées avec succès!")
                plot_forecast(df, forecast)
                
                # Option de téléchargement
                csv = forecast[['ds', 'yhat1']].to_csv(index=False)
                st.download_button("Télécharger les prédictions (CSV)", csv, "predictions.csv", "text/csv", key='download-csv')

if __name__ == "__main__":
    main()