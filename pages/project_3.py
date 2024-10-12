import streamlit as st
import cv2
import numpy as np
from mediapipe import solutions
from tensorflow import keras
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# CSS pour redimensionner les éléments de la caméra et mettre en valeur les résultats
st.markdown("""
    <style>
        .stCamera > div > div > video {
            max-height: 200px !important;
        }
        .stCamera > div > div > div {
            max-height: 200px !important;
        }
        .result-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .emotion-result {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 10px;
        }
        .confidence-result {
            font-size: 18px;
            color: #2ca02c;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Détection d'expression faciale à la demande")

st.write("""
    Comment ça marche ?
    Le modèle compare une photo avec une expression neutre à une photo avec une expression émotionnelle 
    pour prédire l'émotion sur votre visage.

    Le modèle a été construit pendant la semaine Challenge à l'IMT-Nord-Europe, avec 
    Louison Ribouchon et Bastien Dejardin.
""")

VECTOR_SAMPLE_RATIO = 0.7

EXPRESSION_LABELS = {
    0: ("Happy", "🙂"),
    1: ("Fear", "😨"),
    2: ("Surprise", "😮"),
    3: ("Anger", "😠"),
    4: ("Disgust", "🤢"),
    5: ("Sad", "😢")
}

@st.cache_resource
def load_keras_model():
    try:
        model = keras.models.load_model('./models/face_expression_detection.h5', compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

model = load_keras_model()

mp_face_mesh = solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

@st.cache_data
def extract_contours_landmarks(landmarks):
    CONTOURS_INDICES = list(solutions.face_mesh.FACEMESH_CONTOURS)
    CONTOURS_INDICES = np.unique(CONTOURS_INDICES)
    CONTOURS_INDICES = CONTOURS_INDICES[CONTOURS_INDICES < len(landmarks)]
    contours_landmarks = landmarks[CONTOURS_INDICES]
    return contours_landmarks

def process_image(image):
    # Minimise le prétraitement, utilise l'image directement
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    landmark_list = [(landmark.x, landmark.y, landmark.z) for landmark in results.multi_face_landmarks[0].landmark]
    landmark_list_changed = np.array(landmark_list).reshape(len(landmark_list), -1)
    return landmark_list_changed

@st.cache_data
def landmarks_contours_to_vec(neutral, expression):
    X_ = np.array(extract_contours_landmarks(expression - neutral))
    X_ = 1000*X_
    return X_.reshape(1, -1)

def predict_expression(neutral_landmarks, expression_landmarks):
    X = landmarks_contours_to_vec(neutral_landmarks, expression_landmarks)
    if X is not None and model is not None:
        result = model.predict(X, verbose=0)
        predicted_class = np.argmax(result[0])
        confidence = result[0][predicted_class]
        return predicted_class, confidence
    return None, None

def plot_vector_visualization(matrice):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    num_vectors = len(matrice)
    sample_size = int(num_vectors * VECTOR_SAMPLE_RATIO)
    sampled_indices = np.random.choice(num_vectors, sample_size, replace=False)
    
    for i in sampled_indices:
        ax.quiver(0, 0, matrice[i][0], matrice[i][1], scale=1, scale_units='xy', angles='xy', color='b', width=0.002)
    
    ax.set_xlim(np.min(matrice), np.max(matrice))
    ax.set_ylim(np.min(matrice), np.max(matrice))
    ax.grid(True)
    ax.set_title("Vecteurs d'entrée (échantillon)", fontsize=14)
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf


def display_prediction(prediction, confidence):
    if prediction is not None and confidence is not None:
        label, emoji = EXPRESSION_LABELS[prediction]
        st.markdown(f"""
        <div class="result-box">
            <div class="emotion-result">Expression détectée: {label} {emoji}</div>
            <div class="confidence-result">Confiance: {confidence:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("En attente de détection...")


def visualize_landmarks(image, landmarks):
    h, w = image.shape[:2]
    landmark_image = image.copy()
    for landmark in landmarks:
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(landmark_image, (x, y), 2, (0, 255, 0), -1)
    return landmark_image

# Modify the process_image function to return both landmarks and the visualization
def process_image(image):
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None, None
    landmark_list = [(landmark.x, landmark.y, landmark.z) for landmark in results.multi_face_landmarks[0].landmark]
    landmark_list_changed = np.array(landmark_list).reshape(len(landmark_list), -1)
    landmark_visualization = visualize_landmarks(image, landmark_list_changed)
    return landmark_list_changed, landmark_visualization

# Add explanation about vectors and landmarks
st.markdown("""
## Comprendre les vecteurs et les points de repère (landmarks)

Les vecteurs utilisés dans ce modèle sont des représentations mathématiques des mouvements du visage. Chaque vecteur décrit le déplacement d'un point de repère spécifique entre l'expression neutre et l'expression émotionnelle.

Les points de repère (landmarks) sont des points clés sur le visage identifiés par MediaPipe. Ils incluent des points autour des yeux, du nez, de la bouche et du contour du visage. En comparant la position de ces points entre deux expressions, nous pouvons capturer les subtils changements qui caractérisent différentes émotions.
""")


def main():
    col1, col2 = st.columns(2)

    with col1:
        st.write("Prenez une photo avec une expression neutre")
        neutral_photo = st.camera_input("Expression neutre", key="neutral")
    
    with col2:
        st.write("Prenez une photo en exprimant une émotion")
        expression_photo = st.camera_input("Expression émotionnelle", key="expression")

    if neutral_photo is not None and expression_photo is not None:
        col3, col4 = st.columns(2)
        
        with col3:
            st.image(neutral_photo, width=150, caption="Expression neutre")
        with col4:
            st.image(expression_photo, width=150, caption="Expression émotionnelle")

        neutral_image = cv2.imdecode(np.frombuffer(neutral_photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        expression_image = cv2.imdecode(np.frombuffer(expression_photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        neutral_landmarks, neutral_visualization = process_image(neutral_image)
        expression_landmarks, expression_visualization = process_image(expression_image)

        if neutral_landmarks is not None and expression_landmarks is not None:
            st.subheader("Visualisation des points de repère (landmarks)")
            col5, col6 = st.columns(2)
            with col5:
                st.image(neutral_visualization, caption="Points de repère - Expression neutre", use_column_width=True)
            with col6:
                st.image(expression_visualization, caption="Points de repère - Expression émotionnelle", use_column_width=True)

            st.markdown("""
            ### Explication des points de repère extraits
            
            Les points verts sur les images ci-dessus représentent les points de repère détectés par MediaPipe. 
            Nous utilisons spécifiquement les points de contour du visage pour notre analyse. Ces points capturent 
            les changements subtils dans la forme du visage entre l'expression neutre et l'expression émotionnelle.
            
            Le modèle utilise la différence de position de ces points entre les deux expressions pour détecter l'émotion.
            """)

            prediction, confidence = predict_expression(neutral_landmarks, expression_landmarks)
            display_prediction(prediction, confidence)
            
            vector_visualization = plot_vector_visualization(extract_contours_landmarks(expression_landmarks - neutral_landmarks))
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{base64.b64encode(vector_visualization.getvalue()).decode()}" alt="Visualisation des vecteurs" width="600"/>
                </div>
                """, 
                unsafe_allow_html=True
            )


            st.markdown("""
            ### Interprétation des vecteurs
            
            Les flèches bleues dans le graphique ci-dessus représentent les vecteurs de mouvement des points de repère.
            Chaque flèche montre comment un point spécifique s'est déplacé entre l'expression neutre et l'expression émotionnelle.
            La longueur et la direction de ces flèches aident le modèle à identifier l'émotion exprimée.
            """)

            st.markdown("""
            ### Ca ne fonctionne pas bien ? Rien de très étonnant.
            Ce modèle à été dévolppé pour gagner une compétition kaggle, et non pour être déployé en production. Notre avantage sur les autres équipes à été d'utiliser la représentation en vecteur des entrées, ce qui nous a permis d'améliorer :
            - L'information contenue dans les entrées
            - L'explicabilité des résultats (visualisation des vecteurs)
            - réduire la taille des entrées

            ### Pour améliorer, on fait comment ? 
            - une meilleure gestion des données d'entrainement, ici on a rien choisit / controlé. On peut utiliser par exemple des techniques d'augmentation de donnée.
            - le model est dépendant de la position de la tête dans l'image (ex. droite gauche) ce qui ne posait pas de problème pour kaggle ! Mais en production, c'est un problème.
            """)
        else:
            st.error("Impossible de détecter les points de repère sur une ou les deux photos.")

if __name__ == "__main__":
    main()