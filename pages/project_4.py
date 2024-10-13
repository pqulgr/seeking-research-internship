import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import tempfile
import numpy as np
import cv2

def main():
    st.title("Analyse d'image avec YOLOv8n")
    
    # Ajout d'une explication technique concise sur YOLO
    with st.expander("À propos de YOLO"):
        st.markdown("""
### YOLO : Un bond en avant pour la vision par ordinateur

YOLO (*You Only Look Once*), introduit en 2015. Contrairement aux anciennes méthodes comme R-CNN (Region Based Convolutional Neural Networks), qui analysaient des portions d'image, YOLO utilise un réseau de neurones convolutifs pour examiner l'image entière en une seule passe : il ne regarde l'image qu'une seule fois. Résultat : une détection rapide et précise, idéale pour le temps réel.

#### Impact et performances
- **Vitesse** : YOLOv1 atteignait 45 FPS en 2015, avec une précision compétitive.
- **Aujourd'hui** : YOLOv8, la dernière version, peut atteindre **160 FPS** et offre une précision (mAP) allant jusqu’à **57.9%** sur le COCO dataset.

Grâce à ces avancées, YOLOv8 est largement utilisé dans des applications comme la surveillance, la conduite autonome et l’analyse vidéo en temps réel.

        """)
    st.markdown("Déposez une image contenant des objets pour voir si yolo les détecte.")
    # Conteneur principal pour aligner les colonnes
    main_container = st.container()
    
    with main_container:
        col1, col2 = st.columns(2)
        
        with col1:
            # Variables d'état de session
            if "model" not in st.session_state:
                st.session_state["model"] = None
            if "analyzed_image" not in st.session_state:
                st.session_state["analyzed_image"] = None
            if "current_image_path" not in st.session_state:
                st.session_state["current_image_path"] = None

            @st.cache_resource
            def load_model():
                try:
                    model_path = "./models/yolov8n.pt"
                    if os.path.exists(model_path):
                        return YOLO(model_path)
                    else:
                        st.info("Téléchargement du modèle en cours...")
                        return YOLO("yolov8n.pt")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle : {str(e)}")
                    return None

            # Chargement initial du modèle
            if st.session_state["model"] is None:
                st.session_state["model"] = load_model()

            # Options de sélection d'image
            option = st.radio(
                "Choisissez une option",
                ("Télécharger une nouvelle photo", "Utiliser une photo par défaut")
            )

            try:
                if option == "Télécharger une nouvelle photo":
                    uploaded_file = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png"])
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            new_image_path = tmp_file.name
                        
                        if new_image_path != st.session_state["current_image_path"]:
                            st.session_state["analyzed_image"] = None
                            st.session_state["current_image_path"] = new_image_path
                        
                        image = Image.open(uploaded_file)
                    else:
                        st.session_state["current_image_path"] = None
                        st.session_state["analyzed_image"] = None
                        image = None
                else:
                    default_image_path = "assets/images/default_yolo.jpg"
                    if os.path.exists(default_image_path):
                        new_image_path = default_image_path
                        if new_image_path != st.session_state["current_image_path"]:
                            st.session_state["analyzed_image"] = None
                            st.session_state["current_image_path"] = new_image_path
                        image = Image.open(default_image_path)
                    else:
                        st.error("Image par défaut non trouvée.")
                        return

                if image:
                    display_size = (300, 300)
                    image.thumbnail(display_size, Image.Resampling.LANCZOS)
                    st.image(image, caption='Image source', use_column_width=True)

            except Exception as e:
                st.error(f"Erreur lors du traitement de l'image : {str(e)}")
                return

        with col2:
            result_container = st.container()
            
            # Placer le bouton en haut de la colonne
            if st.session_state["current_image_path"] and result_container.button("Analyser"):
                with st.spinner("Analyse en cours..."):
                    if st.session_state["model"] is None:
                        st.error("Le modèle n'a pas pu être chargé.")
                        return
                    
                    results = st.session_state["model"](st.session_state["current_image_path"])
                    result_image = results[0].plot()
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    pil_image = Image.fromarray(result_image_rgb)
                    pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                    
                    st.session_state["analyzed_image"] = np.array(pil_image)
                    
                    # Afficher les objets détectés
                    boxes = results[0].boxes
                    class_counts = {}
                    for box in boxes:
                        class_name = results[0].names[int(box.cls)]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Afficher l'image et les résultats
            if st.session_state["analyzed_image"] is not None:
                result_container.image(st.session_state["analyzed_image"], 
                                      caption='Résultat de l\'analyse', 
                                      use_column_width=True)
                
                result_container.subheader("Objets détectés :")
                for obj, count in class_counts.items():
                    result_container.write(f"- {obj}: {count}")

if __name__ == "__main__":
    main()