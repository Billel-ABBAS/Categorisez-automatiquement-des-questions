from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import parallel_backend
from sklearn.metrics import f1_score, jaccard_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import ast

# Configurer l'URI de suivi pour utiliser un serveur de suivi MLflow distant
mlflow.set_tracking_uri("http://ec2-54-144-47-93.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("tagging_experiment")

# Charger les données nettoyées
df = pd.read_csv('../data/stack_overflow_data_cleaned.csv')

############################### Prétraitement des données #########################
# Combiner tout le contenu des corps de texte nettoyés en une seule chaîne de texte
corpus_combined_title_body = " ".join(df['combined_title_body'].values).lower()

# Tokeniser le texte combiné
corpus_combined_title_body_tokens = corpus_combined_title_body.split()

# Calculer la fréquence de chaque mot dans le corpus
value_counts_combined_title_body = pd.Series(corpus_combined_title_body_tokens).value_counts()

# Créer la liste du vocabulaire des mots les plus fréquents du corpus
vocabulary = list(value_counts_combined_title_body.head(1000).index)

# Vectorisation TF-IDF
vectorizer_supervised = TfidfVectorizer(vocabulary=vocabulary)
X_tfidf = vectorizer_supervised.fit_transform(df['combined_title_body'])

# Enregistrer le TfidfVectorizer dans MLflow
with mlflow.start_run(run_name="vectorizer_supervised"):
    mlflow.sklearn.log_model(vectorizer_supervised, "vectorizer_supervised")

# Convertir les tags de chaînes de caractères en listes
df['split_tags'] = df['split_tags'].apply(ast.literal_eval)

# Combiner tous les tags en une seule liste de corpus
corpus_tags = [tag for sublist in df['split_tags'] for tag in sublist]

# Afficher la fréquence de chaque tag dans le corpus
value_counts_tags = pd.Series(corpus_tags).value_counts()

# Créer la liste du vocabulaire des tags les plus fréquents
vocabulary_tags = list(value_counts_tags.head(1000).index)

# Fonction pour filtrer les tags
def filter_tags(tags):
    return [tag for tag in tags if tag in vocabulary_tags]

# Application de la fonction sur la colonne 'split_tags'
df['split_tags'] = df['split_tags'].apply(filter_tags)

#################################### Partie entraînement ############################################

# Encodage des étiquettes
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['split_tags'])

# Enregistrer le MultiLabelBinarizer dans MLflow
with mlflow.start_run(run_name="mlb"):
    mlflow.sklearn.log_model(mlb, "mlb")

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

models_performance = {}
trained_models = {}

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
    jaccard_scores = []
    for i in range(y_true.shape[0]):
        jaccard_scores.append(jaccard_score(y_true[i], y_pred[i], average='macro'))
    return np.mean(jaccard_scores)

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        print(f'Training {model_name}...')
        model.fit(X_train, y_train)
        
        # Prédire les étiquettes sur les données de test
        y_pred = model.predict(X_test)
        
        # Calculer les métriques de performance
        f1 = f1_score(y_test, y_pred, average='micro')
        jaccard_avg = jaccard(y_test, y_pred)
        
        models_performance[model_name] = {
            'F1 Score': f1,
            'Jaccard Score': jaccard_avg
        }

        print(f'{model_name} - F1 Score: {f1}')
        print(f'{model_name} - Jaccard Score: {jaccard_avg}')
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_metric('F1 Score', f1)
        mlflow.log_metric('Jaccard Score', jaccard_avg)
        
        # Ajouter le modèle et ses performances au dictionnaire des modèles entraînés
        trained_models[model_name] = (model, jaccard_avg)
        
        return model, jaccard_avg

# Liste des modèles à entraîner
model_list = [
    ("Logistic Regression", OneVsRestClassifier(LogisticRegression(max_iter=500), n_jobs=-1)),
    ("SGD Classifier", OneVsRestClassifier(SGDClassifier(max_iter=500), n_jobs=-1)),
    ("Support Vector Machine", OneVsRestClassifier(LinearSVC(max_iter=500), n_jobs=-1)),
    ("Random Forest", OneVsRestClassifier(RandomForestClassifier(n_estimators=30), n_jobs=-1)),
    ("XGBoost", OneVsRestClassifier(XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss'))),
    ("LightGBM", OneVsRestClassifier(LGBMClassifier(n_estimators=500))),
    ("AdaBoost", OneVsRestClassifier(AdaBoostClassifier(n_estimators=30))),
]

# Entraînement des modèles
for model_name, model in model_list:
    with parallel_backend("threading"):
        trained_models[model_name] = train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test)

# Sélection du meilleur modèle
best_model = None
best_jaccard = 0
best_model_name = None

for model_name, (trained_model, jaccard_avg) in trained_models.items():
    if jaccard_avg > best_jaccard:
        best_jaccard = jaccard_avg
        best_model = trained_model
        best_model_name = model_name

print(models_performance)
print(f'Best Model: {best_model_name}, Jaccard Score: {best_jaccard}')

# Enregistrer le meilleur modèle dans MLflow
with mlflow.start_run(run_name="Best_Model"):
    mlflow.sklearn.log_model(best_model, best_model_name)

# Afficher les résultats des performances des modèles
result_df = pd.DataFrame.from_dict(models_performance, orient="index")
print(result_df)


