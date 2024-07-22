import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("Reconnaissance de chiffres manuscrits")
st.write("Dessinez un chiffre et l'application essaiera de le reconnaître.")

# Chargement du modèle
@st.cache_resource
def load_keras_model():
    try:
        model = load_model('./models/project_1.keras')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

model = load_keras_model()

# Prétraitement de l'image
def preprocess_image(image_array):
    # Convertir l'image en niveaux de gris
    gray_image = Image.fromarray(image_array).convert('L')
    # Inverser les couleurs (chiffre blanc sur fond noir)
    inverted_image = ImageOps.invert(gray_image)
    # Redimensionner l'image à 28x28 pixels
    resized_image = np.array(inverted_image.resize((28, 28), Image.LANCZOS))
    # Normaliser les valeurs des pixels
    normalized_image = resized_image.astype('float32') / 255.0
    # Ajouter les dimensions pour correspondre à l'entrée du modèle
    return normalized_image.reshape(1, 28, 28, 1)

# Prédiction
def predict_digit(image_tensor):
    if model is not None:
        predictions = model.predict(image_tensor)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return predicted_digit, confidence
    else:
        return None, None

# Interface utilisateur
st.subheader("Zone de dessin")
st.write("Dessinez un chiffre dans la zone ci-dessous.")

# Paramètres du canvas
stroke_width = 10
stroke_color = "#000000"

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Couleur de remplissage
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Prédire"):
    if canvas_result.image_data is not None:
        image_tensor = preprocess_image(canvas_result.image_data)
        predicted_digit, confidence = predict_digit(image_tensor)
        
        if predicted_digit is not None:
            if confidence<0.6:
                st.write("Chiffre non reconnu")
            else:
                st.write(f'Chiffre prédit : {predicted_digit}')
                st.write(f'Confiance : {confidence:.2%}')
        else:
            st.write("Le modèle n'a pas pu être chargé. Veuillez vérifier le chemin du fichier et l'architecture du modèle.")
    else:
        st.write("Veuillez dessiner un chiffre avant de prédire.")