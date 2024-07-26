import streamlit as st
import cv2
import numpy as np
from mediapipe import solutions
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Détection d'expression faciale à la demande")
st.markdown("Le modèle capture les 5 dernières images pour prédire l'évolution de l'émotion sur votre visage. Ainsi, il n'analyse pas l'image en direct, mais votre réaction au cours du temps.")
st.markdown("Le modèle a été construit pendant la semaine Challenge à l'IMT-Nord-Europe, avec **Louison Ribouchon** et **Bastien Dejardin**.")

CAPTURE_WIDTH = 1280  # Augmentation de la résolution de capture
CAPTURE_HEIGHT = 720
DISPLAY_WIDTH = 640  # Taille d'affichage réduite
DISPLAY_HEIGHT = 480
SKIP_FRAMES = 1
VECTOR_SAMPLE_RATIO = 0.7  # Afficher 70% des vecteurs

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

EXPRESSION_LABELS = ['Happy', 'Fear', 'Surprise', 'Anger', 'Disgust', 'Sad']

mp_face_mesh = solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache_data
def extract_contours_landmarks(landmarks):
    CONTOURS_INDICES = list(solutions.face_mesh.FACEMESH_CONTOURS)
    CONTOURS_INDICES = np.unique(CONTOURS_INDICES)
    CONTOURS_INDICES = CONTOURS_INDICES[CONTOURS_INDICES < len(landmarks)]
    contours_landmarks = landmarks[CONTOURS_INDICES]
    return contours_landmarks

def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmark_list = [(landmark.x, landmark.y, landmark.z) for landmark in results.multi_face_landmarks[0].landmark]
    landmark_list_changed = np.array(landmark_list).reshape(len(landmark_list), -1)
    return landmark_list_changed

@st.cache_data
def landmarks_contours_to_vec(X_temp):
    if len(X_temp) < 1:
        return None
    X_ = np.array(extract_contours_landmarks(X_temp[-1] - X_temp[0]))
    X_ = 1000*X_
    return X_.reshape(1, -1)

def predict_expression(captured_landmarks):
    if len(captured_landmarks) >= 5:
        X = landmarks_contours_to_vec(captured_landmarks[-5:])
        if X is not None and model is not None:
            result = model.predict(X, verbose=0)
            predicted_class = np.argmax(result[0])
            confidence = result[0][predicted_class]
            return predicted_class, confidence
    return None, None

def plot_vector_visualization(matrice):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    origin = np.zeros_like(matrice[0])
    
    num_vectors = len(matrice)
    sample_size = int(num_vectors * VECTOR_SAMPLE_RATIO)
    sampled_indices = np.random.choice(num_vectors, sample_size, replace=False)
    
    for i in sampled_indices:
        ax.quiver(*origin, *matrice[i], scale=1, scale_units='xy', angles='xy', color='b', width=0.002)
    
    ax.set_xlim(np.min(matrice), np.max(matrice))
    ax.set_ylim(np.min(matrice), np.max(matrice))
    ax.grid(True)
    ax.set_title("Vecteurs d'entrée (échantillon)", fontsize=10)
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf

def video_mode():
    captured_landmarks = []
    prediction = None
    confidence = None
    frame_count = 0
    
    video_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    stop_button = st.button("Arrêter la capture")
    
    last_update_time = time.time()
    update_interval = 0.5
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Impossible de lire la vidéo.")
            break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        video_placeholder.image(frame_display, channels="BGR", use_column_width=False)
        
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            landmarks = process_image(frame)
            if landmarks is not None:
                captured_landmarks.append(landmarks)
                
                if len(captured_landmarks) > 5:
                    captured_landmarks.pop(0)
                
                if len(captured_landmarks) == 5:
                    prediction, confidence = predict_expression(captured_landmarks)
                
                if prediction is not None and confidence is not None:
                    expression = EXPRESSION_LABELS[prediction]
                    if confidence > 0.6:
                        prediction_placeholder.write(f"Expression prédite : {expression} (Confiance : {confidence:.2f})")
                    else:
                        prediction_placeholder.write("Expression : incertaine")
            
            last_update_time = current_time
    
    cap.release()

def vector_visualization_mode():
    st.markdown("Ici vous pouvez visualiser ce sur quoi l'algorithme prédit votre expression. Si votre sourire monte, les vecteurs auront tendance à monter, et si vous êtes triste, les vecteurs vont tirer vers le bas.")
    captured_landmarks = []
    frame_count = 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        vector_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    stop_button = st.button("Arrêter la capture")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Impossible de lire la vidéo.")
            break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        landmarks = process_image(frame)
        if landmarks is not None:
            captured_landmarks.append(landmarks)
            
            if len(captured_landmarks) > 5:
                captured_landmarks.pop(0)
            
            if len(captured_landmarks) == 5:
                X_viz = landmarks_contours_to_vec(captured_landmarks)
                img_buf = plot_vector_visualization(X_viz.reshape(-1, 2))
                vector_placeholder.image(img_buf.getvalue(), width=500)
    
    cap.release()

def main():
    mode = st.selectbox("Mode d'affichage", ["Vidéo", "Visualiser l'entrée de l'algorithme"])
    start_button = st.button("Commencer la capture")
    
    if start_button:
        if mode == "Vidéo":
            video_mode()
        else:
            vector_visualization_mode()
    else:
        st.write("Cliquez sur 'Commencer la capture' pour démarrer la capture et la prédiction.")

if __name__ == "__main__":
    main()