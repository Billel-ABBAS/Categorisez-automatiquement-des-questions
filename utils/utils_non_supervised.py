from sklearn.decomposition import LatentDirichletAllocation, NMF  # Pour LDA et NMF
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


# Afficher le taux de remplissage en forme de tableau
def taux_de_Remplissage_tableau(df, affichage_all=False):
    # Calculer le taux de remplissage dans chaque colonne
    filling_rate = round((1 - df.isnull().mean()) * 100, 4)
    filling_rate0 = filling_rate[filling_rate < 100]

    # Créer un nouveau DataFrame avec le taux de remplissage
    filling_info = pd.DataFrame({
        'Colonne': filling_rate.index,
        'Taux_de_Remplissage': filling_rate.values
    })


def vectorizer_transform(text_data, vectorizer):
    if isinstance(text_data, str):
        # Si l'entrée est un seul texte
        bow = vectorizer.transform([text_data])
    else:
        # Si l'entrée est une colonne de DataFrame ou une liste de textes
        bow = vectorizer.transform(text_data)
    return bow


# Calcule les valeurs de cohérence et de perplexité 
def compute_coherence_perplexity(dictionary, corpus, texts, bow, limit, start=2, step=1):
    """
    Calcule les valeurs de cohérence et de perplexité pour différents nombres de topics.
    """
    coherence_values = []  # Liste pour stocker les valeurs de cohérence
    perplexity_values = []  # Liste pour stocker les valeurs de perplexité
    model_list = []  # Liste pour stocker les modèles
    for num_topics in range(start, limit, step):
        # Modèle LDA Gensim pour le score de cohérence
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                num_topics=num_topics,
                                                id2word=dictionary,
                                                random_state=42)
        model_list.append(model)  # Ajout du modèle à la liste
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())  # Ajout de la valeur de cohérence
        
        # Modèle LDA Sklearn pour la perplexité
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(bow)  # Ajustement du modèle
        perplexity_values.append(lda.perplexity(bow))  # Ajout de la valeur de perplexité
    
    return model_list, coherence_values, perplexity_values  # Retourne les modèles et les valeurs calculées



# Visualise les scores de cohérence et de perplexité
def plot_coherence_and_perplexity(coherence_values, perplexity_values, start=2, end=40):
    """
    Visualise les scores de cohérence et de perplexité en fonction du nombre de topics.

    Parameters:
    coherence_values (list): Liste des scores de cohérence.
    perplexity_values (list): Liste des valeurs de perplexité.
    start (int): Valeur de départ pour le nombre de topics (par défaut: 2).
    end (int): Valeur de fin pour le nombre de topics (par défaut: 40).

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Nombre de topics')  # Étiquette pour l'axe x
    ax1.set_ylabel('Coherence score', color=color)  # Étiquette pour l'axe y (cohérence)
    ax1.plot(range(start, end), coherence_values, color=color)  # Trace le score de cohérence
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Crée un deuxième axe partageant le même axe x

    color = 'tab:blue'
    ax2.set_ylabel('Perplexity', color=color)  # Étiquette pour l'axe y (perplexité)
    ax2.plot(range(start, end), perplexity_values, color=color)  # Trace la perplexité
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Ajuste les marges pour éviter que les étiquettes ne soient coupées

    # Ajouter un titre au graphique
    plt.title('Coherence Score et Perplexity par Nombre de Topics')

    # Afficher le graphique
    plt.show()



# Visualise les mots les plus importants pour chaque topic
def plot_top_words(model, feature_names, n_top_words, title, optimal_num_topics):
    """
    Visualise les mots les plus importants pour chaque topic.
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)  # Création de sous-graphiques
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_[:10]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]  # Indices des mots les plus importants
        top_features = [feature_names[i] for i in top_features_ind]  # Noms des mots les plus importants
        weights = topic[top_features_ind]  # Poids des mots

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)  # Graphique en barres horizontales
        ax.set_title(f'Topic {topic_idx+1}', fontdict={'fontsize': 30})
        ax.invert_yaxis()  # Inversion de l'axe y
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        
     # Masquer les sous-graphiques inutilisés
    for ax in axes[optimal_num_topics:]:
        ax.axis('off')

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    


# Visualise les résultats LDA en utilisant t-SNE pour réduire les dimensions à 2
def plot_tsne_lda(lda_output, n_components=2, random_state=42):
    """
    Visualise les résultats LDA en utilisant t-SNE pour réduire les dimensions à 2.

    Parameters:
    lda_output (numpy.ndarray): Les résultats LDA à réduire.
    n_components (int): Le nombre de dimensions pour t-SNE (par défaut: 2).
    random_state (int): L'état aléatoire pour t-SNE (par défaut: 42).

    Returns:
    None
    """
    # Créer un modèle t-SNE pour réduire les dimensions des résultats LDA à 2 composantes
    tsne = TSNE(n_components=n_components, random_state=random_state)

    # Ajuster le modèle t-SNE aux résultats LDA et transformer ces résultats en deux dimensions
    tsne_results = tsne.fit_transform(lda_output)

    # Créer une nouvelle figure avec une taille spécifiée
    plt.figure(figsize=(10, 7))

    # Créer un nuage de points avec les résultats t-SNE
    # Utiliser la première dimension pour l'axe des x et la deuxième dimension pour l'axe des y
    # La couleur des points est déterminée par le topic le plus probable pour chaque document
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=lda_output.argmax(axis=1), cmap='viridis')

    # Ajouter une barre de couleurs pour indiquer à quel topic chaque couleur correspond
    plt.colorbar()

    # Ajouter un titre et des étiquettes aux axes
    plt.title('Visualisation des Topics en 2D avec t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Afficher le graphique
    plt.show()

    
def prediction(new_bow, model, fit_transform = True):
    """
    Prédiction avec LDA ou NMF.

    Args:
    - new_bow: Bag of words du texte à prédire.
    - lda_model: Modèle LDA entraîné.

    Returns:
    - new_topics: Distribution des topics pour le nouveau texte.
    """
    if fit_transform:
        new_topics = model.fit_transform(new_bow)
    else:
        new_topics = model.transform(new_bow)
    return new_topics



# Calcule et filtre les mots basés sur les topics
def calculate_words(new_topics, model):
    """
    Calcule les mots basés sur les topics.

    Args:
    - new_topics: Distribution des topics pour le nouveau texte.
    - model: Modèle LDA OU NMF entraîné.
    
    Returns:
    - new_words: Mots basés sur les topics.
    """
    new_words = np.dot(new_topics, model.components_)
    return new_words

# Filtre les mots
def filter_words(new_words, threshold=0.001):
    """
   filtre les mots basés sur les topics.

    Args:
    - new_words: Mots basés sur les topics.
    - lda_model: Modèle LDA entraîné.
    - threshold: Seuil pour filtrer les mots prédits.

    Returns:
    - new_words_filtered: Mots filtrés en fonction du seuil.
    """
    new_words_filtered = np.where(new_words > threshold, new_words, 0)
    return new_words_filtered


    
# Crée une heatmap pour visualiser la distribution des topics
def plot_heatmap(matrix, title, xlabel='Topics', ylabel='Questions', figsize=(12, 8), cmap='YlOrRd'):
    """
    Crée une heatmap pour visualiser la distribution des topics.

    Parameters:
    matrix (numpy.ndarray or pandas.DataFrame): La matrice de données à visualiser.
    title (str): Le titre du graphique.
    xlabel (str): L'étiquette de l'axe x (par défaut: 'Topics').
    ylabel (str): L'étiquette de l'axe y (par défaut: 'Questions').
    figsize (tuple): La taille de la figure (par défaut: (12, 8)).
    cmap (str): La colormap pour la heatmap (par défaut: 'YlOrRd').

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
   
def coverage_rate(df, actual_column, predicted_column):
    """
    Calcule le taux de couverture entre les tags réels et les tags prédits pour chaque document dans un DataFrame.

    Args:
    - df: Le DataFrame contenant les colonnes des tags réels et prédits.
    - actual_column: Le nom de la colonne contenant les tags réels.
    - predicted_column: Le nom de la colonne contenant les tags prédits.

    Returns:
    - float: Le taux de couverture moyen des tags pour tous les documents.
    """
    def coverage_for_row(row):
        actual_tags = set(row[actual_column])
        predicted_tags = set(row[predicted_column])
        if not actual_tags:
            return 0
        matches = len(actual_tags & predicted_tags)
        total = len(actual_tags)
        return matches / total

    coverage_rates = df.apply(coverage_for_row, axis=1)
    return coverage_rates.mean()


# Calcule la matrice de similarité cosinus entre les topics des questions de test et les questions d'entraînement
def calculate_similarity_matrix(M_test_quest_topics, M_train_quest_topics):
    """
    Calcule la matrice de similarité cosinus entre les topics des questions de test et les questions d'entraînement.

    Args:
    - M_test_quest_topics: Matrice des topics des questions de test.
    - M_train_quest_topics: Matrice des topics des questions d'entraînement.

    Returns:
    - similarity_matrix: Matrice de similarité cosinus.
    """
    similarity_matrix = cosine_similarity(M_test_quest_topics, M_train_quest_topics)  # Matrice de similarité
    return similarity_matrix


################################ Partie teste non supervisé dans le notebook ################################

# Définition de la fonction de prédiction
def predict_keywords(new_text, lda_model, nmf_model, vectorizer, train_topics_lda, train_topics_nmf, tags_train, vectorizer_tags):
    
    # Prétraitement du texte
    new_bow = vectorizer.transform([new_text])
    
    # Prédiction avec LDA
    new_topics_lda = lda_model.transform(new_bow)
    new_words_lda = calculate_words(new_topics_lda, lda_model)
    new_words_filtered_lda = filter_words(new_words_lda, threshold=0)
    
    # Prédiction avec NMF
    new_topics_nmf = nmf_model.transform(new_bow)
    new_words_nmf = calculate_words(new_topics_nmf, nmf_model)
    new_words_filtered_nmf = filter_words(new_words_nmf, threshold=0)
    
    # Prédiction semi-supervisée
    similarity_matrix_lda = calculate_similarity_matrix(new_topics_lda, train_topics_lda)
    semi_supervised_keywords_lda = np.dot(similarity_matrix_lda, tags_train)
    similarity_matrix_nmf = calculate_similarity_matrix(new_topics_nmf, train_topics_nmf)
    semi_supervised_keywords_nmf = np.dot(similarity_matrix_nmf, tags_train)
    
    # Conversion en DataFrame pour visualisation
    words = vectorizer.get_feature_names_out()
    tags = vectorizer_tags.get_feature_names_out()
    
    df_new_words_lda = pd.DataFrame(new_words_filtered_lda, columns=words)
    df_new_semi_supervised_lda = pd.DataFrame(semi_supervised_keywords_lda, columns=tags)
    df_new_words_nmf = pd.DataFrame(new_words_filtered_nmf, columns=words)
    df_new_semi_supervised_nmf = pd.DataFrame(semi_supervised_keywords_nmf, columns=tags)
    
    # Sélection des 5 mots clés les plus pertinents pour chaque prédiction
    predicted_keywords_lda = df_new_words_lda.iloc[0].nlargest(5).index.tolist()
    predicted_semi_supervised_keywords_lda = df_new_semi_supervised_lda.iloc[0].nlargest(5).index.tolist()
    predicted_keywords_nmf = df_new_words_nmf.iloc[0].nlargest(5).index.tolist()
    predicted_semi_supervised_keywords_nmf = df_new_semi_supervised_nmf.iloc[0].nlargest(5).index.tolist()
    
    return predicted_keywords_lda, predicted_semi_supervised_keywords_lda, predicted_keywords_nmf, predicted_semi_supervised_keywords_nmf
