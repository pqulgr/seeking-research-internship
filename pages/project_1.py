import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
import cv2

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
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
    
    # Binarisation de l'image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Trouver les contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Trouver le plus grand contour (supposé être le chiffre)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Extraire le chiffre
        digit = binary_image[y:y+h, x:x+w]
        
        # Ajouter une bordure pour centrer le chiffre
        aspect_ratio = h / w
        if aspect_ratio > 1:
            # Plus haut que large
            new_w = int(h * 1.2)
            new_h = new_w
        else:
            # Plus large que haut
            new_h = int(w * 1.2)
            new_w = new_h
        
        side = max(new_w, new_h)
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        
        padded_digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # Redimensionner l'image à 28x28 pixels
        resized_image = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # Si aucun contour n'est trouvé, retourner une image vide
        resized_image = np.zeros((28, 28), dtype=np.uint8)
    
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


if canvas_result.image_data is not None:
    image_tensor = preprocess_image(canvas_result.image_data)
    predicted_digit, confidence = predict_digit(image_tensor)
        
    if predicted_digit is not None:
        st.write(f'Chiffre prédit : {predicted_digit}')
        st.write(f'Confiance : {confidence:.2%}')
            
        if confidence < 0.7:
            st.write("Attention : La confiance est faible. Essayez de dessiner plus clairement.")
    else:
        st.write("Le modèle n'a pas pu être chargé. Veuillez vérifier le chemin du fichier et l'architecture du modèle.")
else:
    st.write("Veuillez dessiner un chiffre avant de prédire.")

# Affichage de l'image prétraitée (pour le débogage)
if canvas_result.image_data is not None:
    preprocessed = preprocess_image(canvas_result.image_data)
    st.image(preprocessed.reshape(28, 28), caption="Image prétraitée", width=140)