import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import gensim
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split
import pickle

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')  # Télécharger la ressource omw-1.4


import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

# Initialisation des objets NLTK
lemmatizer = WordNetLemmatizer()

# Définir les stop words de spaCy
stop_words = nlp.Defaults.stop_words

# Fonction de traitement de texte global
def process_clean_text(doc, rejoin=True, min_len_word=2):
    def case_normalization(text):
        return text.lower()

    def remove_contractions(text):
        contractions_dict = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
            "'t": " not", "'ve": " have", "'m": " am"
        }
        for contraction, expansion in contractions_dict.items():
            text = re.sub(contraction, expansion, text)
        return text

    def remove_punctuation(text):
        return re.sub(r'[^\w\s#+]', '', text)

    def remove_numbers(text):
        return re.sub(r'\b\d+\b', '', text)

    def remove_extra_spaces(text):
        return re.sub(r'\s+', ' ', text).strip()

    doc = case_normalization(doc)
    doc = remove_contractions(doc)
    doc = remove_punctuation(doc)
    doc = remove_numbers(doc)
    doc = remove_extra_spaces(doc)

    raw_tokens_list = word_tokenize(doc)
    raw_tokens_list = ['c#' if token == 'c' else token for token in raw_tokens_list]

    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]

    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1]
        if tag in {'NN', 'NNP', 'NNS'}:
            return tag
        return None

    filtered_tokens = [w for w in more_than_N if get_wordnet_pos(w) in {'NN', 'NNP', 'NNS'}]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

    if rejoin:
        return " ".join(lemmatized_tokens)
    return lemmatized_tokens

# Fonction pour nettoyer le texte HTML en supprimant les URLs, les balises `style`, `script` et `code`
def clean_html(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    soup = BeautifulSoup(text, 'lxml')
    for element in soup(['style', 'script', 'code']):
        element.extract()
    cleaned_text = soup.get_text(separator=' ')
    return cleaned_text

def vectorizer_transform(text_data, vectorizer):
    if isinstance(text_data, str):
        bow = vectorizer.transform([text_data])
    else:
        bow = vectorizer.transform(text_data)
    return bow

def calculate_words(new_topics, model):
    new_words = np.dot(new_topics, model.components_)
    return new_words

def filter_words(new_words, threshold=0.001):
    new_words_filtered = np.where(new_words > threshold, new_words, 0)
    return new_words_filtered

def calculate_similarity_matrix(M_test_quest_topics, M_train_quest_topics):
    similarity_matrix = cosine_similarity(M_test_quest_topics, M_train_quest_topics)
    return similarity_matrix

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

    file_path_tags = '../Model/unsupervised/tags_train.npz'
    with np.load(file_path_tags) as data:
        tags_train = data['tags_train']

    return lda_model, nmf_model, vectorizer, vectorizer_tags, train_topics_lda, train_topics_nmf, tags_train

def predict_keywords(new_text, lda_model, nmf_model, vectorizer, train_topics_lda, train_topics_nmf, tags_train, vectorizer_tags):
    new_bow = vectorizer.transform([new_text])
    
    new_topics_lda = lda_model.transform(new_bow)
    new_words_lda = calculate_words(new_topics_lda, lda_model)
    new_words_filtered_lda = filter_words(new_words_lda, threshold=0)
    
    new_topics_nmf = nmf_model.transform(new_bow)
    new_words_nmf = calculate_words(new_topics_nmf, nmf_model)
    new_words_filtered_nmf = filter_words(new_words_nmf, threshold=0)
    
    similarity_matrix_lda = calculate_similarity_matrix(new_topics_lda, train_topics_lda)
    semi_supervised_keywords_lda = np.dot(similarity_matrix_lda, tags_train)
    similarity_matrix_nmf = calculate_similarity_matrix(new_topics_nmf, train_topics_nmf)
    semi_supervised_keywords_nmf = np.dot(similarity_matrix_nmf, tags_train)
    
    words = vectorizer.get_feature_names_out()
    tags = vectorizer_tags.get_feature_names_out()
    
    df_new_words_lda = pd.DataFrame(new_words_filtered_lda, columns=words)
    df_new_semi_supervised_lda = pd.DataFrame(semi_supervised_keywords_lda, columns=tags)
    df_new_words_nmf = pd.DataFrame(new_words_filtered_nmf, columns=words)
    df_new_semi_supervised_nmf = pd.DataFrame(semi_supervised_keywords_nmf, columns=tags)
    
    predicted_keywords_lda = df_new_words_lda.iloc[0].nlargest(5).index.tolist()
    predicted_semi_supervised_keywords_lda = df_new_semi_supervised_lda.iloc[0].nlargest(5).index.tolist()
    predicted_keywords_nmf = df_new_words_nmf.iloc[0].nlargest(5).index.tolist()
    predicted_semi_supervised_keywords_nmf = df_new_semi_supervised_nmf.iloc[0].nlargest(5).index.tolist()
    
    return predicted_keywords_lda, predicted_semi_supervised_keywords_lda, predicted_keywords_nmf, predicted_semi_supervised_keywords_nmf
