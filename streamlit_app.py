import streamlit as st
from PIL import Image
import os

# Configuration de la page
st.set_page_config(page_title="Mon Portfolio", layout="wide")

# Titre principal
st.title("Bienvenue sur mon Portfolio")

# Description
st.write("Découvrez mes projets ci-dessous. Cliquez sur une carte pour explorer le projet.")
st.write("🟢 Fast = 1-2 min pour explorer, have fun !")
st.write("🟠 Medium = 3-6 min. Temps de gestion des modèles ou plus d'option à découvrire.")
st.write("🔴 Long > 10 min, Il faut se concentrer par ici, mais c'est interessant ! ")
# Liste de projets avec exploration time ajoutée
projects = [
    {
        "title": "Reconnaître des chiffres écrit",
        "description": "Le fameux 'Hello world' du deep learning, la reconnaissance des chiffres manuscrits est un classique. Ici, j'intégre le modèle dans une application web.",
        "image": "assets/images/mnist_p1.png",
        "page": "project_1",
        "explore_time": "Fast"
    },
    {
        "title": "Ségrégation spontannée",
        "description": "Dans les années 1970, Thomas Schelling, économiste américain, énonce qu'une préférence pour le choix de ses voisins sur un critère arbitraire conduit à une ségrégation totale bien que cette ségrégation ne corresponde pas aux préférences individuelles. En 2024, a-t-il toujours raison ?",
        "image": "assets/images/parabole_p2.png",
        "page": "project_2",
        "explore_time": "Long"
    },
    {
        "title": "Reconnaissance d'émotion",
        "description": "Les ordinateurs sont-ils capables de comprendre nos émotions ? Non, mais ils peuvent les détecter",
        "image": "assets/images/FER.png",
        "page": "project_3",
        "explore_time": "Medium"
    },
    {
        "title": "Analyse d'image avec YOLOv8",
        "description": "Découvrez comment YOLOv8, un modèle de détection d'objets, peut être utilisé pour analyser des images en temps réel.",
        "image": "assets/images/YOLO.png",
        "page": "project_4",
        "explore_time": "Fast"
    },
    {
        "title": "Prédiction de Séries Temporelles",
        "description": "Explorez la puissance de NeuralProphet pour prédire l'évolution de séries temporelles. Déposez vos données pour tester le modèle.",
        "image": "assets/images/neural_prophet.png",
        "page": "project_5",
        "explore_time": "Fast"
    }
]

# Fonction de génération des couleurs
def get_color_label(explore_time):
    if explore_time == "Fast":
        return "🟢 Fast"
    elif explore_time == "Medium":
        return "🟠 Medium"
    elif explore_time == "Long":
        return "🔴 Long"

# Modification de la fonction project_card
def project_card(title, description, image, page, explore_time):
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                img = Image.open(image)
                aspect_ratio = img.size[1] / img.size[0]
                standard_width = 300
                standard_height = int(standard_width * aspect_ratio)
                st.image(image, width=standard_width)
            except FileNotFoundError:
                st.error(f"Image non trouvée : {image}")
                
        with col2:
            # Ajouter l'indicateur de couleur à côté du titre
            st.subheader(f"{title} ({get_color_label(explore_time)})")
            st.write(description)
            if st.button("Explorer le projet", key=f"btn_{page}"):
                st.switch_page(f"pages/{page}.py")
        st.markdown("---")

# Affichage des projets
for project in projects:
    project_card(**project)

# Instructions pour la configuration des fichiers
if not os.path.exists("pages"):
    st.warning("""
    Pour que ce portfolio fonctionne correctement, assurez-vous de :
    1. Créer un dossier 'pages' dans le même répertoire que ce script
    2. Créer les fichiers de projet :
       - project_1.py
       - project_2.py
       - project_3.py
       - project_4.py
       - project_5.py
    3. Les placer dans le dossier 'pages'
    4. Vérifier que vos images sont bien dans le chemin spécifié
    """)