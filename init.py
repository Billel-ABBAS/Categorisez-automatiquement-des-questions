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

# Charger le modèle spaCy en anglais
import en_core_web_sm
nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
nlp = en_core_web_sm.load()

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
    
    

################# Partie non_supervisée ##########################
from sklearn.decomposition import LatentDirichletAllocation, NMF  # Pour LDA et NMF
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE




def tfidf_vectorizer_transform(text_data, tfidf_vectorizer):
    if isinstance(text_data, str):
        # Si l'entrée est un seul texte
        bow = tfidf_vectorizer.transform([text_data])
    else:
        # Si l'entrée est une colonne de DataFrame ou une liste de textes
        bow = tfidf_vectorizer.transform(text_data)
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

    
def lda_prediction(new_bow, lda_model):
    """
    Prédiction avec LDA.

    Args:
    - new_bow: Bag of words du texte à prédire.
    - lda_model: Modèle LDA entraîné.

    Returns:
    - new_topics: Distribution des topics pour le nouveau texte.
    """
    new_topics = lda_model.transform(new_bow)
    return new_topics



# Calcule et filtre les mots basés sur les topics
def calculate_words(new_topics, lda_model):
    """
    Calcule les mots basés sur les topics.

    Args:
    - new_topics: Distribution des topics pour le nouveau texte.
    - lda_model: Modèle LDA entraîné.
    
    Returns:
    - new_words: Mots basés sur les topics.
    """
    new_words = np.dot(new_topics, lda_model.components_)
    return new_words

# Filtre les mots
def filter_words(new_words, lda_model, threshold=0.01):
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
    
   
# Calcule le taux de couverture entre les tags prédits et les tags réels 
def coverage_rate(predicted_tags, actual_tags):
    """
    Calcule le taux de couverture entre les tags prédits et les tags réels.
    """
    matches = np.sum((predicted_tags > 0) & (actual_tags > 0), axis=1)  # Comptage des correspondances
    total = np.sum(actual_tags > 0, axis=1)  # Comptage des tags réels
    return np.mean(matches / total)  # Calcul du taux de couverture moyen



# Définition de la fonction de prédiction
def predict_keywords(new_text, lda_model, tfidf_vectorizer, train_topics, tags_train, vectorizer_tags, threshold=0.01):
    # Prétraitement du texte
    new_bow = tfidf_vectorizer_transform([new_text], tfidf_vectorizer)
    
    # Prédiction avec LDA
    new_topics = lda_prediction(new_bow, lda_model)
    new_words = calculate_words(new_topics, lda_model)
    new_words_filtered = filter_words(new_words, lda_model, threshold=0.01)
    
    # Prédiction semi-supervisée
    similarity_matrix = cosine_similarity(new_topics, train_topics)
    semi_supervised_keywords = np.dot(similarity_matrix, tags_train)
    
    # Conversion en DataFrame pour visualisation
    words = tfidf_vectorizer.get_feature_names_out()
    tags = vectorizer_tags.get_feature_names_out()
    
    df_new_words = pd.DataFrame(new_words_filtered, columns=words)
    df_new_semi_supervised = pd.DataFrame(semi_supervised_keywords, columns=tags)
    
    # Sélection des 5 mots clés les plus pertinents pour chaque prédiction
    predicted_keywords = df_new_words.iloc[0].nlargest(5).index.tolist()
    predicted_semi_supervised_keywords = df_new_semi_supervised.iloc[0].nlargest(5).index.tolist()
    
    return predicted_keywords, predicted_semi_supervised_keywords