import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from neuralprophet import NeuralProphet
import numpy as np
from datetime import datetime, timedelta

class ExportAnalyzer:
    def __init__(self):
        self.df = None
        self.model = None
        self.forecast = None
        self.data_loaded = False
        self.model_trained = False
        self.future_periods = 300

    def load_data(self, uploaded_file, date_column, value_column):
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                raise ValueError("Format de fichier non supporté")
            
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.rename(columns={date_column: 'ds', value_column: 'y'})
            self.df = df[['ds', 'y']]
            self.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {str(e)}")
            self.data_loaded = False
            return False

    def generate_data(self):
        START_DATE = datetime(2020, 1, 1)
        NUM_DAYS = 1000
        BASE_EXPORT = 1000
        TREND = 0.1
        SEASONALITY = 0.2
        NOISE = 0.05

        dates = [START_DATE + timedelta(days=i) for i in range(NUM_DAYS)]
        exports = [max(0, int(BASE_EXPORT + TREND * i + SEASONALITY * BASE_EXPORT * np.sin(2 * np.pi * i / 365) + NOISE * BASE_EXPORT * np.random.randn())) for i in range(NUM_DAYS)]
        
        self.df = pd.DataFrame({'ds': dates, 'y': exports})
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

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ExportAnalyzer()

    data_source = st.radio("Choisissez la source des données :", ("Charger un fichier", "Générer des données"))

    if data_source == "Charger un fichier":
        uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Aperçu des données:")
            st.write(df_preview.head())

            columns = df_preview.columns.tolist()
            date_column = st.selectbox("Sélectionnez la colonne de date", columns)
            value_column = st.selectbox("Sélectionnez la colonne de valeur", [col for col in columns if col != date_column])

            if st.button("Charger les données"):
                if st.session_state.analyzer.load_data(uploaded_file, date_column, value_column):
                    st.success("Données chargées avec succès!")
                    st.plotly_chart(st.session_state.analyzer.plot_input_data())
    else:
        if st.button("Générer des données"):
            st.session_state.analyzer.generate_data()
            st.success("Données générées avec succès!")
            st.plotly_chart(st.session_state.analyzer.plot_input_data())

    if st.session_state.analyzer.data_loaded:
        if st.button("Entraîner le modèle et prédire"):
            with st.spinner("Entraînement du modèle en cours..."):
                st.session_state.analyzer.train_model()
            st.success("Modèle entraîné avec succès!")
            st.plotly_chart(st.session_state.analyzer.plot_forecast())

if __name__ == "__main__":
    main()