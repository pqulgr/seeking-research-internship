import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Mon Portfolio", layout="wide")

# Titre principal
st.title("Bienvenue sur mon Portfolio")

# Description
st.write("Découvrez mes projets ci-dessous. Cliquez sur une carte pour en savoir plus.")

# Liste de projets factices
projects = [
    {
        "title": "Projet 1",
        "description": "Description courte du projet 1",
        "image": "assets/images/parabole_p1.png",
        "link": "projet1"
    }
]

def project_card(title, description, image, link):
    st.image(image, use_column_width=True)
    st.subheader(title)
    st.write(description)
    if st.button("En savoir plus", key=title):
        st.write(f"Vous avez cliqué sur {title}. Lien vers: {link}")
        # Note: Dans une vraie application, vous redirigeriez vers la page du projet ici

# Affichage des projets en grille
cols = st.columns(3)
for i, project in enumerate(projects):
    with cols[i % 3]:
        project_card(**project)