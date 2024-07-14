import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_map_pop(n,tx_vide):
    M = np.random.choice([1, 2], size=(n, n))
    n_empty = int(n*tx_vide*n)
    indices = np.random.choice(n * n, size=n_empty, replace=False)
    places_libres = []
    for index in indices:
        M[index // n, index % n] = 0
        places_libres.append((index // n, index % n))
    return M, places_libres
def get_wrapped_neighbors(i, j, n):
    """Obtenir les indices des voisins avec repliement."""
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di != 0 or dj != 0:  # Exclure la cellule elle-même
                ni = (i + di) % n
                nj = (j + dj) % n
                neighbors.append((ni, nj))
    return neighbors

def create_map_insatisfaction(map_, n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if map_[i, j] != 0:
                same_type = 0
                total_neighbors = 0
                neighbors = get_wrapped_neighbors(i, j, n)
                for ni, nj in neighbors:
                    if map_[ni, nj] != 0:
                        total_neighbors += 1
                        if map_[ni, nj] == map_[i, j]:
                            same_type += 1
                if total_neighbors > 0:
                    M[i, j] = 1 - (same_type / total_neighbors)
    return M

def create_place_libre(map_pop, n):
    l=[]
    for i in range(n):
        for j in range(n):
            if map_pop[i,j]==0:
                l.append([i,j])
    return l

class population():
    def __init__(self, n_pop, taux_insatisfaction, taux_vide):
        self.n_pop = n_pop
        self.taux_insatisfaction = taux_insatisfaction
        self.taux_vide = taux_vide
        self.map_pop, self.places_libres = create_map_pop(n_pop, taux_vide)
        self.map_insatisfaction = create_map_insatisfaction(self.map_pop, n_pop)


def main():
    st.set_page_config(layout="wide")
    st.title("Modélisation de population et évolution de la ségrégation")
    
    st.markdown("""
    ## Introduction
    
    La ségrégation du territoire par évolution spontanée des groupes mixtes est un phénomène complexe et potentiellement dangereux. 
    Ce projet vise à explorer différents modèles de ségrégation, en commençant par le célèbre modèle de Thomas Schelling, 
    puis en proposant des alternatives plus proches de la réalité.
    
    ### Objectifs du projet :
    1. Implémenter et analyser le modèle de Schelling
    2. Développer un modèle alternatif continu
    3. Créer un modèle intégrant des réseaux d'individus
    4. Comparer les résultats des différents modèles
    
    Explorez chaque modèle ci-dessous pour comprendre leur fonctionnement et leurs implications.
    """)
    
    st.header("1. Modèle de Schelling")
    st.markdown("""
    Le modèle de Schelling, proposé en 1971, démontre comment des préférences individuelles modestes 
    pour avoir des voisins similaires peuvent mener à une ségrégation importante au niveau global.
    
    #### Fonctionnement :
    - Les individus sont répartis sur une grille
    - Chaque case peut être occupée par un individu de type A, de type B, ou être vide
    - Un individu est satisfait si une certaine proportion de ses voisins est du même type que lui
    - Les individus insatisfaits se déplacent vers des cases vides jusqu'à ce que tous soient satisfaits
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Taille de la grille", 10, 100, 50, key="schelling_n")
        vide = st.slider("Proportion de cases vides", 0.1, 0.5, 0.1, key="schelling_vide")
    with col2:
        insatisfaction = st.slider("Seuil d'insatisfaction", 0.0, 1.0, 1/3, key="schelling_insatisfaction")
    
    if st.button("Génerer une population"):
        st.markdown("### Explication de la population")
        st.markdown("""
            La population est composée de 3 choses:
            - 0 emplacement vides
            - 1 individus de type 1
            - 2 individus de type 2
            """)
        st.markdown("""Chaque individu a un score d'insatisfaction, calculé comme suit :""")
        st.latex(r"\text{Score d'insatisfaction} = \sum_{i=1}^{8} \frac{\text{voisins du même type que l'individu}}{\text{total de voisins présents}}")
        pop = population(n, insatisfaction, vide)
        
        # Créer les heatmaps avec Seaborn
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 ligne, 2 colonnes

        sns.heatmap(pop.map_pop, ax=axes[0], cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=False)
        cbar = axes[0].collections[0].colorbar
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels([0, 1, 2])
        axes[0].set_title("Types des individus")
        #axes[1].set_xlabel("X Axis Label")
        #axes[1].set_ylabel("Y Axis Label")

        sns.heatmap(pop.map_insatisfaction, ax=axes[1], cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=False)
        axes[1].set_title("Carte de l'insatisfaction")
        
        # Ajuster l'espace entre les subplots
        plt.tight_layout()
        
        # Afficher les heatmaps dans Streamlit
        st.pyplot(fig)
    
    st.header("2. Modèle continu")
    st.markdown("""
    Ce modèle étend le concept de Schelling à un espace continu en deux dimensions, 
    offrant une représentation plus réaliste de l'espace urbain.
    
    #### Fonctionnement :
    - Les individus sont placés aléatoirement sur un plan
    - Le voisinage est défini par une distance plutôt que par des cases adjacentes
    - Les individus se déplacent en fonction de la composition de leur voisinage
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_2d = st.slider("Nombre d'individus", 100, 2000, 1000, key="2d_n")
        distance = st.slider("Distance de voisinage", 20, 100, 60, key="2d_distance")
    with col2:
        insatisfaction_2d = st.slider("Seuil d'insatisfaction", 0.0, 1.0, 1/3, key="2d_insatisfaction")
    
    if st.button("Simuler le modèle 2D"):
        print("oka")
        # Simulation du modèle 2D
        # ...
    
    st.header("3. Modèle continu avec réseau d'amis")
    st.markdown("""
    Ce modèle intègre la notion de réseaux sociaux, ajoutant une dimension supplémentaire à la dynamique de ségrégation.
    
    #### Fonctionnement :
    - Les individus sont placés sur un plan comme dans le modèle 2D
    - Chaque individu a un réseau d'amis défini par un rayon
    - Les déplacements sont influencés à la fois par le voisinage spatial et le réseau d'amis
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_3 = st.slider("Nombre d'individus", 100, 2000, 1000, key="3_n")
        rayon_ami = st.slider("Rayon des amis", 20, 200, 100, key="3_rayon")
    with col2:
        distance_3 = st.slider("Distance de voisinage", 20, 100, 60, key="3_distance")
        insatisfaction_3 = st.slider("Seuil d'insatisfaction", 0.0, 1.0, 1/3, key="3_insatisfaction")
    
    if st.button("Simuler le modèle avec réseau d'amis"):
        print("ok3")
        # Simulation du modèle avec réseau d'amis
        # ...
    
    st.header("Comparaison des modèles")
    st.markdown("""
    La comparaison des résultats des différents modèles nous permet de mieux comprendre 
    les mécanismes de la ségrégation et l'impact des différentes hypothèses sur le phénomène.
    
    Voici un graphique comparatif des taux de ségrégation obtenus avec les trois modèles :
    """)
    
    # Ici, vous pouvez ajouter un graphique comparatif des résultats des trois modèles
    # Par exemple, un graphique montrant l'évolution de la ségrégation en fonction du seuil d'insatisfaction
    # pour chaque modèle
    
    st.header("Conclusion")
    st.markdown("""
    Cette étude nous a permis d'explorer différentes approches de modélisation de la ségrégation spatiale.
    Nous avons pu observer que :
    
    1. Le modèle de Schelling montre comment des préférences individuelles modestes peuvent mener à une ségrégation importante.
    2. Le modèle 2D offre une représentation plus réaliste de l'espace urbain et permet d'explorer l'impact de la distance sur la ségrégation.
    3. L'introduction de réseaux sociaux dans le troisième modèle ajoute une dimension supplémentaire à la dynamique de ségrégation.
    
    Ces modèles nous aident à mieux comprendre les mécanismes de la ségrégation et peuvent servir de base 
    pour développer des politiques urbaines plus inclusives.
    """)

if __name__ == "__main__":
    main()