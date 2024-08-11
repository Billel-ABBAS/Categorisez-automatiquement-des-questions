from sklearn.metrics import jaccard_score
from joblib import parallel_backend, dump
from sklearn.metrics import f1_score
import joblib
import numpy as np
import pandas as pd

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

# Définir la fonction jaccard pour calculer le score de Jaccard moyen
def jaccard(y_true, y_pred):
    """
    Calcule le score de Jaccard moyen pour des prédictions multi-étiquettes.

    Parameters:
    y_true (numpy.ndarray): Les étiquettes réelles.
    y_pred (numpy.ndarray): Les étiquettes prédites.

    Returns:
    float: Le score de Jaccard moyen.
    """
    # Initialiser une liste pour stocker les scores de Jaccard individuels
    jaccard_scores = []
    
    # Boucler sur chaque ensemble d'étiquettes réelles et prédites
    for i in range(y_true.shape[0]):
        # Calculer le score de Jaccard pour l'échantillon i
        jaccard_scores.append(jaccard_score(y_true[i], y_pred[i], average='macro'))
    
    # Retourner la moyenne des scores de Jaccard
    return np.mean(jaccard_scores)


# Initialiser les dictionnaires pour stocker les performances des modèles et les modèles entraînés
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test, models_performance):
    """
    Entraîne un modèle, évalue ses performances et enregistre le modèle.

    Parameters:
    model_name (str): Le nom du modèle.
    model (sklearn model): L'instance du modèle à entraîner.
    X_train (sparse matrix): Les données d'entraînement.
    y_train (array): Les étiquettes d'entraînement.
    X_test (sparse matrix): Les données de test.
    y_test (array): Les étiquettes de test.

    Returns:
    model (sklearn model): Le modèle entraîné.
    jaccard_avg (float): Le score moyen de Jaccard du modèle.
    """
    # Utiliser le backend parallèle avec threading
    
    print(f'Training {model_name}...')

    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédire les étiquettes sur les données de test
    y_pred = model.predict(X_test)
    
    # Calculer le score F1
    f1 = f1_score(y_test, y_pred, average='micro')
    
    # Calculer le score moyen de Jaccard
    jaccard_avg = jaccard(y_test, y_pred)
    
    # Stocker les performances du modèle
    models_performance[model_name] = {
        'F1 Score': f1,
        'Jaccard Score': jaccard_avg
    }

    print(f'{model_name} - F1 Score: {f1}')
    print(f'{model_name} - Jaccard Score: {jaccard_avg}')
    
    # # Enregistrer le modèle dans un fichier
    # model_path = f'Model/supervised/{model_name.replace(" ", "_").lower()}.pkl'
    # dump(model, model_path)
    # print(f'{model_name} saved to {model_path}')
    
    # Retourner le modèle entraîné et le score moyen de Jaccard
    return model, jaccard_avg


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



# Calcule le taux de couverture entre les tags réels et les tags prédits
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