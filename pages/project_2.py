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

def create_example_grid(neighbors):
    return np.array(neighbors).reshape(3, 3)

def plot_grids(population, insatisfaction, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    sns.heatmap(population, ax=ax1, cmap="YlGnBu", cbar=False, square=True, linewidths=0.5, linecolor="gray", annot=True, fmt='d')
    ax1.set_title("Population")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    sns.heatmap(insatisfaction, ax=ax2, cmap="YlOrRd", cbar=False, square=True, linewidths=0.5, linecolor="gray", annot=True, fmt='.2f')
    ax2.set_title("Insatisfaction")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

class Population:
    def __init__(self, n_pop, taux_insatisfaction, taux_vide):
        self.n_pop = n_pop
        self.taux_insatisfaction = taux_insatisfaction
        self.taux_vide = taux_vide
        self.map_pop, self.places_libres = create_map_pop(n_pop, taux_vide)
        self.map_insatisfaction = create_map_insatisfaction(self.map_pop, n_pop)
        self.iteration = 0

    def simulate(self, iter):
        n = 0
        indices_i, indices_j = np.where(self.map_insatisfaction > self.taux_insatisfaction)
        indices_i, indices_j = list(indices_i), list(indices_j)
        progress_bar = st.progress(0)
        while n < iter and len(indices_i) > 0:
            progress_bar.progress((n + 1) / iter)
            choice_from = np.random.randint(0, len(indices_i))
            choice_to = self.places_libres[np.random.randint(0, len(self.places_libres))]
            i, j = indices_i.pop(choice_from), indices_j.pop(choice_from)
            self.map_pop[choice_to[0], choice_to[1]] = self.map_pop[i, j]
            self.map_pop[i, j] = 0
            self.map_insatisfaction = create_map_insatisfaction(self.map_pop, self.n_pop)
            indices_i, indices_j = np.where(self.map_insatisfaction > self.taux_insatisfaction)
            indices_i, indices_j = list(indices_i), list(indices_j)
            n += 1
        self.iteration += n
        progress_bar.empty()
        return self

    def plot_map(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(self.map_pop, ax=axes[0], cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=False)
        cbar = axes[0].collections[0].colorbar
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Vide', 'Type 1', 'Type 2'])
        axes[0].set_title("Types des individus")
        sns.heatmap(self.map_insatisfaction, ax=axes[1], cmap="YlOrRd", cbar=True, xticklabels=False, yticklabels=False)
        axes[1].set_title("Carte de l'insatisfaction")
        plt.tight_layout()
        return fig
    


def main_schelling():

    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'population' not in st.session_state:
        st.session_state.population = None

    st.markdown("""
    ## Introduction
    
    Dans les années 1970, Thomas Schelling, economiste américain, énonce qu’une préférence pour le choix de ses voisins sur un critère arbitraire conduit à une ségrégation totale 
    bien que cette ségrégation ne corresponde pas aux préférences individuelles. En 2024, a-t-il toujours raison ?
        
    
    ### Fonctionnement du modèle :
    - Les individus sont répartis sur une grille
    - Chaque case peut être occupée par un individu de type 1, de type 2, ou être vide
    - Un individu est satisfait si une certaine proportion de ses voisins est du même type que lui
    - Les individus insatisfaits se déplacent vers des cases vides jusqu'à ce que tous soient satisfaits ou que le nombre maximal d'itérations soit atteint
    
    Suivez les étapes ci-dessous pour explorer le modèle de Schelling.
    """)

    # Étape 1 : Configuration initiale
    st.header("Étape 1 : Configuration initiale")
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Taille de la grille", 10, 100, 30)
    with col2:
        vide = st.slider("Proportion de cases vides", 0.1, 0.5, 0.1, step=0.05)

    if st.button("Générer une population") or st.session_state.step >= 1:
        st.session_state.step = max(st.session_state.step, 1)
        st.session_state.population = Population(n, 1, vide)  # Taux d'insatisfaction initial à 1
        st.markdown("### Population initiale")
        st.markdown("""Chaque individu a un score d'insatisfaction, calculé comme suit :""")
        st.latex(r"\text{Score d'insatisfaction} = \sum_{i=1}^{8} \frac{\text{voisins du même type que l'individu}}{\text{total de voisins présents}}")
        st.pyplot(st.session_state.population.plot_map())
        # Calcul et affichage des statistiques
        total_cells = st.session_state.population.n_pop ** 2
        occupied_cells = np.sum(st.session_state.population.map_pop != 0)
        type_1_cells = np.sum(st.session_state.population.map_pop == 1)
        type_2_cells = np.sum(st.session_state.population.map_pop == 2)
            
        st.markdown("### Statistiques")
        col1, col2, col3 = st.columns(3)
        col1.metric("Taux d'occupation", f"{occupied_cells/total_cells:.2%}")
        col2.metric("Proportion Type 1", f"{type_1_cells/occupied_cells:.2%}")
        col3.metric("Proportion Type 2", f"{type_2_cells/occupied_cells:.2%}")
        st.markdown("""
        **Explication :**
        - Cases bleues foncées (2) : Individus de type 2
        - Cases bleues claires (1) : Individus de type 1
        - Cases blanches (0) : Emplacements vides
        
        La carte de droite montre l'insatisfaction initiale de chaque individu.
        """)

    # Étape 2 : Définition du seuil d'insatisfaction
    if st.session_state.step >= 1:
        st.header("Étape 2 : Définition du seuil d'insatisfaction")
        st.markdown("""
        Le seuil d'insatisfaction détermine à partir de quel pourcentage de voisins différents un individu décide de déménager.
        Par exemple, un seuil de 0.33 signifie que l'orsque 1/3 de ses voisins sont différents, l'individu déménage.
        1/3 est l'hypothèse posée par T.Schelling lors de son article.
        """)

        col1, col2, col3 = st.columns(3)
        dump = np.zeros((3,3))
        with col1:
            st.markdown("### Cas 1: Satisfait")
            grid1 = create_example_grid([1, 1, 1, 1, 1, 1, 0, 1, 0])
            insatisfaction1 = create_map_insatisfaction(grid1, 3)
            insatisfaction1_ = dump.copy()
            insatisfaction1_[1,1] = insatisfaction1[1,1]
            insatisfaction1 = insatisfaction1_.copy()
            fig1 = plot_grids(grid1, insatisfaction1, "Majoritairement entouré de même type")
            st.pyplot(fig1)
            st.write(f"Insatisfaction: {insatisfaction1[1, 1]:.2f}")
            
        with col2:
            st.markdown("### Cas 2: Insatisfait")
            grid2 = create_example_grid([2, 2, 2, 1, 1, 2, 0, 2, 0])
            insatisfaction2 = create_map_insatisfaction(grid2, 3)
            insatisfaction2_ = dump.copy()
            insatisfaction2_[1,1] = insatisfaction2[1,1]
            insatisfaction2 = insatisfaction2_.copy()
            fig2 = plot_grids(grid2, insatisfaction2, "Majoritairement entouré de type différent")
            st.pyplot(fig2)
            st.write(f"Insatisfaction: {insatisfaction2[1, 1]:.2f}")
            
        with col3:
            st.markdown("### Cas 3: Cas limite")
            grid3 = create_example_grid([1, 2, 1, 2, 1, 1, 0, 2, 0])
            insatisfaction3 = create_map_insatisfaction(grid3, 3)
            insatisfaction3_ = dump.copy()
            insatisfaction3_[1,1] = insatisfaction3[1,1]
            insatisfaction3 = insatisfaction3_.copy()
            fig3 = plot_grids(grid3, insatisfaction3, "Entouré de manière équilibrée")
            st.pyplot(fig3)
            st.write(f"Insatisfaction: {insatisfaction3[1, 1]:.2f}")

        insatisfaction = st.slider("Seuil d'insatisfaction toléré", 0.0, 1.0, 1/3, step=0.01)
        
        if st.button("Appliquer le seuil d'insatisfaction") or st.session_state.step >= 2:
            st.session_state.step = max(st.session_state.step, 2)
            st.session_state.population.taux_insatisfaction = insatisfaction
            st.session_state.population.map_insatisfaction = create_map_insatisfaction(st.session_state.population.map_pop, st.session_state.population.n_pop)



    # Étape 3 : Simulation
    if st.session_state.step >= 2:
        st.header("Étape 3 : Simulation")
        st.markdown("""
        Maintenant que nous avons défini notre population et le seuil d'insatisfaction, nous pouvons lancer la simulation.
        Les individus insatisfaits vont se déplacer vers des cases vides jusqu'à ce que tous soient satisfaits ou que le nombre maximal d'itérations soit atteint.
        """)
        iterations = st.number_input("Nombre d'itérations", min_value=1, max_value=10000, value=1000, step=100)
        
        if st.button("Lancer la simulation"):
            total_cells = st.session_state.population.n_pop ** 2
            unsatisfied_cells_before = np.sum(st.session_state.population.map_insatisfaction > st.session_state.population.taux_insatisfaction)

            st.session_state.population = st.session_state.population.simulate(iterations)
            segreg_cells = np.sum(st.session_state.population.map_insatisfaction==0)
            st.markdown(f"### Résultats après {st.session_state.population.iteration} itérations")
            st.pyplot(st.session_state.population.plot_map())
            
            # Calcul et affichage des statistiques
            unsatisfied_cells_after = np.sum(st.session_state.population.map_insatisfaction > st.session_state.population.taux_insatisfaction)
            
            st.markdown("### Statistiques finales")
            col1, col2, col3 = st.columns(3)
            col1.metric("Taux d'insatisfaction avant déplacement des individus", f"{unsatisfied_cells_before/occupied_cells:.2%}")
            col2.metric("Taux d'insatisfaction après déplacement des individus", f"{unsatisfied_cells_after/occupied_cells:.2%}")
            col3.metric("Pourcentages d'individus entourés uniquement du même type",f"{((segreg_cells - total_cells)/occupied_cells)+1:.2%}")

            st.markdown("On observe ainsi, qu'en voulant 1/3 de ses voisins semblables, un très fort pourcentage de la population n'est finalement entourée que de ses semblables, résultant ainsi sur la population totale une forte ségrégation. C'est ce que T.Schelling à observé comme effet pervers. Un faible critère de séléction pour ses voisins mène la population vers une forte ségrégation, bien que celle-ci ne soit pas voulu par les individus.")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Simulation de Ségrégation")
    st.title("Modélisation de population et évolution de la ségrégation")
    option = st.selectbox("Sélectionnez le chapitre", ("Chapitre 1 : Modèle de T.Schelling", "Chapitre 2 : Modèle continu", "Modèle 3 : Modèle continu avec liens"))
    if option=="Chapitre 1 : Modèle de T.Schelling":
        main_schelling()
    elif option == "Chapitre 2 : Modèle continu":
        st.write("In progress, aviaible soon")
    elif option =="Modèle 3 : Modèle continu avec liens":
        st.write("In progress, aviaible soon")
    else:
        st.write("Non existant")