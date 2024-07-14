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
        "title": "Reconnaître des chiffres écrit",
        "description": "Le fameux 'Hello world' du deep learning, la reconnaissance des chiffres manuscrits est un classique. Ici, j'intégre le modèle dans une application web.",
        "image": "assets/images/mnist_p1.png",
        "link": "projet1"
    },
    {
        "title": "Ségrégation spontannée",
        "description": "Dans les années 1970, Thomas Schelling, economiste américain, énonce qu’une préférence pour le choix de ses voisins sur un critère arbitraire conduit à une ségrégation totale bien que cette ségrégation ne corresponde pas aux préférences individuelles. En 2024, a-t-il toujours raison ?",
        "image": "assets/images/parabole_p2.png",
        "link": "projet2"
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