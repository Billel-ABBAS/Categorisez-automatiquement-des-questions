import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from IPython.display import display
import nltk
from nltk import pos_tag, word_tokenize
import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')  # Télécharger la ressource omw-1.4

import spacy
nlp = spacy.load("en_core_web_sm")

# # Charger le modèle spaCy en anglais
# import en_core_web_sm
# nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
# nlp = en_core_web_sm.load()

# Initialisation des objets NLTK
lemmatizer = WordNetLemmatizer()

# Définir les stop words de spaCy
stop_words = nlp.Defaults.stop_words

################# Partie exploratoire ##########################

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

    # Afficher le DataFrame avec les informations sur le taux de remplissage
    if affichage_all:
        # Afficher toutes les colonnes avec leur taux de remplissage
        filling_info = filling_info.reset_index(drop=True).sort_values(by="Taux_de_Remplissage", ascending=False).reset_index(drop=True)
    else:
        # Afficher uniquement les colonnes avec des valeurs manquantes
        filling_info = filling_info[filling_info['Taux_de_Remplissage'] < 100].reset_index(drop=True).sort_values(by="Taux_de_Remplissage", ascending=False).reset_index(drop=True)
    display(filling_info)

# Fonction pour diviser les tags en utilisant '><' et '-', supprimer les points '.' et les chiffres seuls
def split_tags(tags):
    # Enlever les balises initiales et finales '<' et '>'
    tags = tags.strip('<>').lower()
    # Diviser en utilisant '><' puis diviser chaque segment par '-' et supprimer les points '.'
    split_tags = [subtag.replace('.', '') for tag in tags.split('><') for subtag in tag.split('-')]
    # Supprimer les chiffres seuls
    split_tags = [tag for tag in split_tags if not tag.isdigit() and tag != '3x']
    return split_tags

# Afficher les informations sur les tokens
def display_tokens_infos(tokens):
    """Affiche les informations sur le corpus de tokens."""
    print(f"Nombre de tokens: {len(tokens)}")
    print(f"Nombre de tokens uniques: {len(set(tokens))}\n")
    print("Exemple de tokens:", tokens[:30], "\n")

# Générer et afficher un word cloud pour une liste de tags
def generate_wordcloud(corpus_tags):
    combined_tags = ' '.join(corpus_tags)
    wordcloud_tags = WordCloud(width=800, height=400, background_color='white').generate(combined_tags)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_tags, interpolation='bilinear')
    plt.title('Word Cloud des Split Tags')
    plt.axis('off')
    plt.show()

# Compter les tags et visualiser les tags les plus fréquents
def visualize_top_tags(corpus_tags, top_n=20):
    tag_counts = Counter(corpus_tags)
    common_tags = tag_counts.most_common(top_n)
    tags, counts = zip(*common_tags)
    plt.figure(figsize=(14, 7))
    sns.barplot(x=list(counts), y=list(tags))
    plt.title(f'Top {top_n} des tags les plus fréquents')
    plt.xlabel('Counts')
    plt.ylabel('Tags')
    plt.show()

# Analyser les catégories grammaticales des tags uniques
def analyze_pos_tags(unique_tags, threshold=0.03):
    tags_with_pos = pos_tag(unique_tags)
    pos_counts = Counter(tag for _, tag in tags_with_pos)
    total_count = sum(pos_counts.values())
    other_count = sum(count for tag, count in pos_counts.items() if (count / total_count) < threshold)
    filtered_pos_counts = {tag: count for tag, count in pos_counts.items() if (count / total_count) >= threshold}
    if other_count > 0:
        filtered_pos_counts['Autres'] = other_count
    print("Tags avec catégories grammaticales :", tags_with_pos[:50])
    print("---------------------------- \n")
    print("Comptage des catégories grammaticales :", pos_counts)
    print("-----------------------------\n")
    print("Comptage des catégories grammaticales filtrées :", filtered_pos_counts)
    return filtered_pos_counts

# Créer un diagramme en secteurs (pie chart) pour visualiser la répartition des catégories grammaticales
def plot_pos_pie_chart(pos_counts):
    labels = pos_counts.keys()
    sizes = pos_counts.values()
    explode = [0.05] * len(labels)
    plt.figure(figsize=(10, 10))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=260, explode=explode, textprops={'fontsize': 15})
    plt.axis('equal')  # Assure que le pie chart est dessiné comme un cercle
    plt.title("Répartition des Catégories Grammaticales", fontsize=18)
    plt.legend(labels, loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

# Fonction de traitement de texte global
def process_clean_text(doc, rejoin=True, min_len_word=2):
    """
    Fonction de traitement de texte global

    Arguments :
    ----------------------
    doc : str : le document (c'est-à-dire un texte au format str) à traiter

    rejoin : bool : si True, retourne une chaîne de caractères sinon retourne la liste des tokens
    min_len_word : int : la longueur minimale des mots à ne pas exclure

    Les noms, voire certains verbes, permettent de définir ces mots clés. Les adjectifs ou les adverbes sont beaucoup moins pertinents.

    Retour :
    -------
    Une chaîne de caractères (si rejoin est True) ou une liste de tokens
    """
    
    # Fonction pour normaliser les cas
    def case_normalization(text):
        return text.lower()

    # Fonction pour supprimer les contractions anglaises (modifiée pour développer les contractions)
    def remove_contractions(text):
        contractions_dict = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am"
        }
        for contraction, expansion in contractions_dict.items():
            text = re.sub(contraction, expansion, text)
        return text

    # Fonction pour supprimer la ponctuation (conserver certains caractères)
    def remove_punctuation(text):
        return re.sub(r'[^\w\s#+]', '', text)

    # Nouvelle fonction pour supprimer uniquement les chiffres seuls
    def remove_numbers(text):
        return re.sub(r'\b\d+\b', '', text)

    # Fonction pour supprimer les espaces supplémentaires
    def remove_extra_spaces(text):
        return re.sub(r'\s+', ' ', text).strip()

    # Mise en minuscule, suppression des espaces en trop, des liens et des balises HTML et de code
    doc = case_normalization(doc)
    doc = remove_contractions(doc)
    doc = remove_punctuation(doc)
    doc = remove_numbers(doc)
    doc = remove_extra_spaces(doc)

    # Tokenisation avec word_tokenize
    raw_tokens_list = word_tokenize(doc)
    raw_tokens_list = ['c#' if token == 'c' else token for token in raw_tokens_list]

    # Suppression des stop words avec spaCy
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # Suppression des mots de longueur inférieure à min_len_word
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]

    # Fonction pour obtenir la catégorie grammaticale de WordNet
    def get_wordnet_pos(word):
        """Retourne la catégorie grammaticale pour la lemmatisation."""
        tag = pos_tag([word])[0][1]
        if tag in {'NN', 'NNP', 'NNS'}:
            return tag
        return None

    filtered_tokens = [w for w in more_than_N if get_wordnet_pos(w) in {'NN', 'NNP', 'NNS'}]

    # Lemmatisation
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

    # Retour du résultat
    if rejoin:
        return " ".join(lemmatized_tokens)

    return lemmatized_tokens


# Fonction pour nettoyer le texte HTML en supprimant les URLs, les balises `style`, `script` et `code`
def clean_html(text):
    """
    Nettoyer les balises HTML, supprimer les liens et récupérer le texte brut, 
    tout en conservant les tags spécifiques dans les balises <code>.
    
    Args:
        text (str): Texte brut avec HTML.
             
    Returns:
        str: Texte nettoyé.
    """
    
    # Suppression des URLs du texte HTML brut
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Utilisation du parser lxml pour une analyse plus robuste
    soup = BeautifulSoup(text, 'lxml')

    # Suppression des balises <style> et <script>
    for element in soup(['style', 'script', 'code']):
        element.extract()

    # Récupération du texte nettoyé incluant le contenu des balises <code>
    cleaned_text = soup.get_text(separator=' ')

    return cleaned_text



# Créér Histogrammes des variables numériques
def plot_histograms(df, variables, bins=50, figsize=(18, 6)):
    """
    Crée une figure avec des histogrammes pour les variables spécifiées.

    Arguments:
    ----------
    df : DataFrame : Le DataFrame contenant les données.
    variables : list : Liste des noms des colonnes pour lesquelles tracer les histogrammes.
    bins : int : Nombre de bins pour les histogrammes. Par défaut, 50.
    figsize : tuple : Taille de la figure. Par défaut, (18, 6).
    """
    # Création d'une figure avec 1 ligne et len(variables) colonnes
    fig, axes = plt.subplots(1, len(variables), figsize=figsize)

    # Boucle pour tracer chaque histogramme dans une sous-figure
    for ax, var in zip(axes, variables):
        sns.histplot(df[var], bins=bins, kde=False, ax=ax)
        ax.set_title(f'Distribution des {var}')
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    

# Boîte à moustaches:
def plot_boxplots(df, variables, y_limits, figsize=(18, 6)):
    """
    Crée une figure avec des boîtes à moustaches pour les variables spécifiées.

    Arguments:
    ----------
    df : DataFrame : Le DataFrame contenant les données.
    variables : list : Liste des noms des colonnes pour lesquelles tracer les boîtes à moustaches.
    y_limits : dict : Dictionnaire des limites de l'axe y pour chaque variable.
    figsize : tuple : Taille de la figure. Par défaut, (18, 6).
    """
    # Création d'une figure avec 1 ligne et len(variables) colonnes
    fig, axes = plt.subplots(1, len(variables), figsize=figsize)

    # Boucle pour tracer chaque boîte à moustaches dans une sous-figure
    for ax, var in zip(axes, variables):
        sns.boxplot(data=df[var], ax=ax)
        ax.set_title(f'Boîte à moustaches des {var}')
        ax.set_ylim(y_limits[var])  # Définir les limites de l'axe y

    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    
# Matrice de corrélation    
def plot_correlation_matrix(df, variables, figsize=(10, 8), cmap='coolwarm'):
    """
    Crée une figure avec une matrice de corrélation pour les variables spécifiées.

    Arguments:
    ----------
    df : DataFrame : Le DataFrame contenant les données.
    variables : list : Liste des noms des colonnes pour lesquelles tracer la matrice de corrélation.
    figsize : tuple : Taille de la figure. Par défaut, (10, 8).
    cmap : str : Colormap utilisée pour la heatmap. Par défaut, 'coolwarm'.
    """
    # Création de la figure
    plt.figure(figsize=figsize)
    
    # Calcul de la matrice de corrélation
    corr_matrix = df[variables].corr()
    
    # Tracer la heatmap
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.title('Matrice de corrélation')
    
    # Afficher la figure
    plt.show()
