import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup



# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des objets NLTK
lemmatizer = WordNetLemmatizer()

# Définir les stop words de NLTK
stop_words = set(stopwords.words('english'))

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



# Fonction de prédiction des tags pour un nouveau texte
def predict_tags(text, model, tfidf_vectorizer, mlb, top_n=5):
    """
    Prédit les tags pour un texte donné en utilisant un modèle entraîné, un vectoriseur TF-IDF et un MultiLabelBinarizer.

    Parameters:
    text (str): Le texte à analyser.
    model (sklearn model): Le modèle de machine learning entraîné.
    tfidf_vectorizer (TfidfVectorizer): Le vectoriseur TF-IDF entraîné.
    mlb (MultiLabelBinarizer): L'encodeur multi-label entraîné.
    top_n (int): Le nombre de tags les plus probables à retourner.

    Returns:
    list: Les tags prédits pour le texte donné.
    """
    
    # Transformer le texte en vecteur TF-IDF
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Prédire les probabilités des tags pour le texte donné
    predicted_probs = model.predict_proba(text_tfidf)
        
    # Sélectionner les indices des top_n probabilités les plus élevées
    top_indices = np.argsort(predicted_probs, axis=1)[:, -top_n:]
    
    # Convertir les indices des top_n probabilités en tags
    top_tags = [ [mlb.classes_[i] for i in indices] for indices in top_indices ]
    
    # Retourner les tags prédits pour le texte donné
    return top_tags[0] if top_tags else []
