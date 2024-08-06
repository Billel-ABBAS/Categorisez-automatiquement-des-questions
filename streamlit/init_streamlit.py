import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import LatentDirichletAllocation, NMF  # Pour LDA et NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fonction de transformation du vectorizer
def vectorizer_transform(text_data, vectorizer):
    if isinstance(text_data, str):
        # Si l'entrée est un seul texte
        bow = vectorizer.transform([text_data])
    else:
        # Si l'entrée est une colonne de DataFrame ou une liste de textes
        bow = vectorizer.transform(text_data)
    return bow

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

# Fonction pour charger les modèles et les données
def load_models_and_data():
    with open('../Model/unsupervised/lda_model.pkl', 'rb') as file:
        lda_model = pickle.load(file)

    with open('../Model/unsupervised/nmf_model.pkl', 'rb') as file:
        nmf_model = pickle.load(file)

    with open('../Model/unsupervised/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('../Model/unsupervised/vectorizer_tags.pkl', 'rb') as file:
        vectorizer_tags = pickle.load(file)

    with open('../Model/unsupervised/train_topics_lda.pkl', 'rb') as file:
        train_topics_lda = pickle.load(file)

    with open('../Model/unsupervised/train_topics_nmf.pkl', 'rb') as file:
        train_topics_nmf = pickle.load(file)

    # Charger la matrice des tags depuis le fichier compressé NPZ
    file_path_tags = '../Model/unsupervised/tags_train.npz'
    with np.load(file_path_tags) as data:
        tags_train = data['tags_train']

    return lda_model, nmf_model, vectorizer, vectorizer_tags, train_topics_lda, train_topics_nmf, tags_train

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
