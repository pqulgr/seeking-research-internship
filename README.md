# 📦 Streamlit App Starter Kit 
```
⬆️ (Replace above with your app's name)
```

Description of the app ...

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-model-builder-template.streamlit.app/)

## Section Heading

This is filler text, please replace this with text for this section.

## Further Reading

This is filler text, please replace this with a explanatory text about further relevant resources for this repo
- Resource 1
- Resource 2
- Resource 3


        
        
        
    
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
