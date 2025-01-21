import os
import sys
import pytest
import pandas as pd
import numpy as np
import re
from collections import Counter
from bs4 import BeautifulSoup
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
import mlflow

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des fonctions utilitaires
from utils.utils_exploratoire import (
    taux_de_Remplissage_tableau, 
    split_tags, 
    display_tokens_infos, 
    analyze_pos_tags,  
    process_clean_text, 
    stop_words, 
    clean_html
)
from utils.utils_non_supervised import (
    vectorizer_transform, 
    filter_words, 
    coverage_rate
)
from utils.utils_supervised import (
    predict_tags, 
    jaccard
)

### Partie exploratoire ###

########################## Teste de la fonction taux_de_Remplissage_tableau ############################

def test_taux_de_Remplissage_tableau():
    # Création d'un DataFrame de test
    data = {
        "col1": [1, 2, 3, None],
        "col2": [None, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    taux_de_Remplissage_tableau(df)

########################## Teste de la fonction split_tags ############################

test_data = [
    ("<python><data-science><3x><machine.learning>", ['python', 'data', 'science', 'machinelearning']),
    ("<html><css><3><12.3>", ['html', 'css']),
    ("<full-stack><web-development>", ['full', 'stack', 'web', 'development']),
    ("", [''])
]

@pytest.mark.parametrize("input_tags, expected_output", test_data)
def test_split_tags(input_tags, expected_output):
    assert split_tags(input_tags) == expected_output

########################## Teste de la fonction analyze_pos_tags ############################

def test_analyze_pos_tags(capfd):
    unique_tags = ["dog", "bark", "cat", "meow", "programming", "coding", "test", "testing", "analysis", "analyzing"]
    expected_counts = {
        'NN': 6,  # Supposons que 6 de ces mots sont des noms communs
        'VBG': 4  # Supposons que 4 sont des verbes au participe présent
        
    }
    # Appel de la fonction
    result = analyze_pos_tags(unique_tags, threshold=0.1)  # Ajustez le seuil pour correspondre aux données attendues

    # Capture et test de la sortie imprimée
    out, err = capfd.readouterr()
    assert "Tags avec catégories grammaticales :" in out
    assert "Comptage des catégories grammaticales :" in out
    assert "Comptage des catégories grammaticales filtrées :" in out

    # Test de la sortie de la fonction
    assert result == expected_counts, f"Expected {expected_counts}, but got {result}"

########################## Teste de la fonction process_clean_text ############################

@pytest.mark.parametrize("doc, rejoin, expected_result", [
  # Test de normalisation en minuscules (case_normalization)
  ("UPPERCASE WORDS and MiXeD CaSe", True, "uppercase word case"),
  
  # Test de suppression des contractions (remove_contractions)
  ("I can't do it, won't you help?", True, "help"),
  
  # Test de suppression de la ponctuation (remove_punctuation)
  ("Hello, world! How are you?", True, "hello world"),
  
  # Test de conservation de certains caractères spéciaux
  ("Use #include <iostream> in C++", True, "use include iostream c++"),
  
  # Test de suppression des chiffres seuls (remove_numbers)
  ("Error in script.js at line 10", True, "error scriptjs line"),
  
  # Test de conservation des mots composés avec des chiffres
  ("Update to Python3.9 and Node.js 14", True, "update python39 nodejs"),
  
  # Test de suppression des espaces supplémentaires (remove_extra_spaces)
  ("Too   many    spaces    here", True, "space"),
  
  # Test de tokenisation et traitement des cas spéciaux (comme 'c#')
  ("Learn C# and ASP.NET", True, "learn c# aspnet"),
  
  # Test de suppression des stop words
  ("The quick brown fox jumps over the lazy dog", True, "quick brown fox jump lazy dog"),
  
  # Test de longueur minimale des mots (min_len_word)
  ("A B CD EF GHI", True, "cd ef ghi"),
  
  # Test de filtrage par catégorie grammaticale et lemmatisation
  ("The cats are running quickly", True, "cat"),
  
  # Test avec rejoin=False
  ("Updating node.js and npm packages", False, ["nodejs", "npm", "package"]),
  
  # Test complet combinant plusieurs aspects
  ("Learn Python, Java, and C++ today!", True, "learn python java c++ today"),
])
def test_process_clean_text(doc, rejoin, expected_result):
  result = process_clean_text(doc, rejoin)
  assert result == expected_result, f"Expected {expected_result}, but got {result}"

########################## Teste de la fonction clean_html ############################

@pytest.mark.parametrize("input_html, expected_output", [
  # Test de suppression des URLs
  ("<p>Visit <a href='http://example.com'>here</a></p>", "Visit"),
  
  # Test de suppression des balises HTML et script
  ("<div>Text <script>alert('Hello');</script>with style</div>", "Text  with style"),
  
  # Test de suppression des balises style
  ("<style>.class { color: red; }</style>Styled text", "Styled text"),
  
  # Test de conservation du contenu des balises code
  ("Hello <code>Code snippet</code> world", "Hello   world"),
  
  # Test de gestion d'un mélange complexe
  ("<div>Mixed <a href='http://test.com'>content</a> with <code>code</code></div>",
   "Mixed"),
])
def test_clean_html(input_html, expected_output):
  result = clean_html(input_html)
  assert result.strip() == expected_output.strip(), f"Expected '{expected_output}', but got '{result}'"

### Partie non supervisé ###

########################## Teste de la fonction vectorizer_transform ############################

@pytest.fixture
def setup_tfidf_vectorizer():
    # Création d'un vectoriseur TF-IDF pour les tests
    vectorizer = TfidfVectorizer()
    data = ["hello world", "test data", "data science is cool"]
    vectorizer.fit(data)  # Préparation du vectoriseur avec quelques données
    return vectorizer

@pytest.mark.parametrize("input_data, expected_shape", [
    ("hello world", (1, 7)),  # Test avec une seule chaîne
    (["hello world", "test data"], (2, 7)),  # Test avec une liste de chaînes
    (["hello world", "test data", "data science is cool"], (3,7))
])
def test_vectorizer_transform(input_data, expected_shape, setup_tfidf_vectorizer):
    vectorizer = setup_tfidf_vectorizer
    bow = vectorizer_transform(input_data, vectorizer)
    assert bow.shape == expected_shape, f"Expected shape {expected_shape}, but got {bow.shape}"

########################## Teste de la fonction filter_words ############################

@pytest.mark.parametrize("new_words, threshold, expected_result", [
    (np.array([0.002, 0.0005, 0.1, 0.05]), 0.001, np.array([0.002, 0, 0.1, 0.05])),  # Cas où certains mots sont en dessous du seuil
    (np.array([0.001, 0.001, 0.001]), 0.001, np.array([0, 0, 0])),  # Cas où tous les mots sont égaux au seuil
    (np.array([0.0001, 0.0002]), 0.001, np.array([0, 0])),  # Cas où tous les mots sont en dessous du seuil
    (np.array([]), 0.001, np.array([])),  # Cas d'entrée vide
])
def test_filter_words(new_words, threshold, expected_result):
    result = filter_words(new_words, threshold)
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

########################## Teste de la fonction coverage_rate  ############################

@pytest.fixture
def tags_df():
    data = {
        'actual_tags': [['python', 'data'], ['machine', 'learning'], ['data', 'science'], []],
        'predicted_tags': [['python', 'science'], ['machine'], ['data'], ['python']]
    }
    return pd.DataFrame(data)

def test_coverage_rate(tags_df):
    # Calcul du taux de couverture attendu
    # Première ligne: 1/2 = 0.5 (intersection = {python})
    # Deuxième ligne: 1/2 = 0.5 (intersection = {machine})
    # Troisième ligne: 1/2 = 0.5 (intersection = {data})
    # Quatrième ligne: 0 (intersection = set(), pas de tags réels)
    expected_coverage_rate = (0.5 + 0.5 + 0.5 + 0) / 4
    
    # Appel de la fonction
    actual_coverage_rate = coverage_rate(tags_df, 'actual_tags', 'predicted_tags')
    
    assert actual_coverage_rate == expected_coverage_rate, f"Expected coverage rate to be {expected_coverage_rate}, but got {actual_coverage_rate}"

### Partie supervisé ###
                                   
########################## Teste de la fonction jaccard  ############################

@pytest.mark.parametrize("y_true, y_pred, expected_score", [
    (np.array([[1, 0, 1], [1, 1, 0]]), np.array([[1, 0, 0], [1, 0, 0]]), 0.5),  # Cas simple
    (np.array([[0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 0], [0, 0, 1]]), 1.0),        # Correspondance parfaite
    (np.array([[1, 1, 1], [0, 0, 0]]), np.array([[0, 0, 0], [1, 1, 1]]), 0.0),        # Aucune correspondance
])
def test_jaccard(y_true, y_pred, expected_score):
    result = jaccard(y_true, y_pred)
    np.testing.assert_almost_equal(result, expected_score, decimal=7, err_msg=f"Expected {expected_score}, but got {result}")

########################## Teste de la fonction predict_tags ############################

# Récupérer les identifiants AWS depuis les variables d'environnement configurées dans Github
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configurer MLflow pour utiliser un chemin compatible avec WSL
mlflow.set_tracking_uri("http://ec2-54-144-47-93.compute-1.amazonaws.com:5000/")

def load_mlflow_model(model_uri, model_type='sklearn'):
    try:
        if model_type == 'sklearn':
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError("Type de modèle non supporté : {}".format(model_type))
    except Exception as e:
        # Affichage de l'erreur dans la console
        print(f"Erreur lors du chargement du modèle {model_uri}: {str(e)}")
        return None

# Fixture pour charger les modèles et artefacts MLflow
@pytest.fixture
def setup_mlflow_models():
    # Charger le TfidfVectorizer depuis MLflow
    logged_vectorizer = 'runs:/e614fbf9d9104f128aabef80e261cc51/vectorizer_supervised'
    tfidf_vectorizer = load_mlflow_model(logged_vectorizer, 'sklearn')

    # Charger le MultiLabelBinarizer depuis MLflow
    logged_mlb = 'runs:/e01623a6cb0e4330819fc438b28e03ba/mlb'
    mlb = load_mlflow_model(logged_mlb, 'sklearn')

    # Charger le modèle XGBoost en tant que Sklearn Model depuis MLflow
    logged_model = 'runs:/1bf16235c76f47fda1a1d5f5355994af/XGBoost'
    model = load_mlflow_model(logged_model, 'sklearn')

    return model, tfidf_vectorizer, mlb

# Exemple de texte à prédire
@pytest.fixture
def example_text():
    return '''
    'java socket binding port bind socket client port code connection choose port exception case exception connect love pointer'
    '''

# exemple de teste :
def test_predict_tags(setup_mlflow_models, example_text):
    model, tfidf_vectorizer, mlb = setup_mlflow_models
    
    # Appeler la fonction pour prédire les tags
    predicted_tags = predict_tags(example_text, model, tfidf_vectorizer, mlb, top_n=5)
    
    # Exemple de tags attendus (cela dépend du modèle entraîné et des données)
    expected_tags = ['java', 'client', 'sockets']
    
    # Vérifier que les tags prédits contiennent au moins certains des tags attendus
    assert all(tag in predicted_tags for tag in expected_tags), \
        f"Expected some of the tags {expected_tags} to be in {predicted_tags}"

