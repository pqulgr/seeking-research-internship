import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from neuralprophet import NeuralProphet
import numpy as np
from datetime import datetime, timedelta
from collections.abc import Mapping

class ExportAnalyzer:
    def __init__(self):
        self.df = None
        self.model = None
        self.forecast = None
        self.data_loaded = False
        self.model_trained = False
        self.future_periods = 300

    def load_data(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            df['ds'] = pd.to_datetime(df['ds'])
            self.df = df[['ds', 'y']]
            self.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {str(e)}")
            self.data_loaded = False
            return False

    def generate_data(self, num_points):
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(num_points)]
        values = np.random.randint(50, 200, num_points) + np.sin(np.arange(num_points) / 20) * 50
        self.df = pd.DataFrame({'ds': dates, 'y': values})
        self.data_loaded = True

    def plot_input_data(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df['ds'], y=self.df['y'], mode='lines', name='Données d\'entrée'))
        fig.update_layout(title="Données d'entrée", xaxis_title="Date", yaxis_title="Valeur")
        return fig

    def train_model(self):
        self.model = NeuralProphet()
        metrics = self.model.fit(self.df, freq='D')
        future = self.model.make_future_dataframe(self.df, periods=self.future_periods)
        self.forecast = self.model.predict(future)
        self.model_trained = True
        return metrics

    def plot_forecast(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df['ds'], y=self.df['y'], mode='lines', name='Données réelles'))
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['yhat1'], mode='lines', name='Prédictions'))
        fig.update_layout(title="Prédictions vs Réalité", xaxis_title="Date", yaxis_title="Valeur")
        return fig

def main():
    st.title("Prédiction de Séries Temporelles avec NeuralProphet")

    analyzer = ExportAnalyzer()

    data_source = st.radio("Choisissez la source des données :", ("Charger un fichier", "Générer des données"))

    if data_source == "Charger un fichier":
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            if analyzer.load_data(uploaded_file):
                st.success("Données chargées avec succès!")
                st.plotly_chart(analyzer.plot_input_data())
    else:
        num_points = st.slider("Nombre de points à générer", min_value=100, max_value=1000, value=500)
        if st.button("Générer des données"):
            analyzer.generate_data(num_points)
            st.success("Données générées avec succès!")
            st.plotly_chart(analyzer.plot_input_data())

    if analyzer.data_loaded:
        if st.button("Entraîner le modèle et prédire"):
            with st.spinner("Entraînement du modèle en cours..."):
                analyzer.train_model()
            st.success("Modèle entraîné avec succès!")
            st.plotly_chart(analyzer.plot_forecast())

if __name__ == "__main__":
    main()