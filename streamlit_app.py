import streamlit as st
from PIL import Image
import os

# Configuration de la page
st.set_page_config(page_title="Mon Portfolio", layout="wide")

# Titre principal
st.title("Bienvenue sur mon Portfolio")

# Description
st.write("Découvrez mes projets ci-dessous. Cliquez sur une carte pour explorer le projet.")

# Liste de projets avec liens et images standardisées
projects = [
    {
        "title": "Reconnaître des chiffres écrit",
        "description": "Le fameux 'Hello world' du deep learning, la reconnaissance des chiffres manuscrits est un classique. Ici, j'intégre le modèle dans une application web.",
        "image": "assets/images/mnist_p1.png",
        "page": "project_1"
    },
    {
        "title": "Ségrégation spontannée",
        "description": "Dans les années 1970, Thomas Schelling, economiste américain, énonce qu'une préférence pour le choix de ses voisins sur un critère arbitraire conduit à une ségrégation totale bien que cette ségrégation ne corresponde pas aux préférences individuelles. En 2024, a-t-il toujours raison ?",
        "image": "assets/images/parabole_p2.png",
        "page": "project_2"
    },
    {
        "title": "Reconnaissance d'émotion",
        "description": "Les ordinateurs sont-ils capables de comprendre nos émotions ? (Non, mais ils peuvent les détécter)",
        "image": "assets/images/FER.png",
        "page": "project_3"
    }
]

def project_card(title, description, image, page):
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            # Vérifier si l'image existe et la charger de manière standardisée
            try:
                img = Image.open(image)
                aspect_ratio = img.size[1] / img.size[0]
                standard_width = 300
                standard_height = int(standard_width * aspect_ratio)
                st.image(image, width=standard_width)
            except FileNotFoundError:
                st.error(f"Image non trouvée : {image}")
                
        with col2:
            st.subheader(title)
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
    2. Renommer vos fichiers de projet en :
       - project_1.py
       - project_2.py
       - project_3.py
    3. Les placer dans le dossier 'pages'
    4. Vérifier que vos images sont bien dans le chemin spécifié
    """)