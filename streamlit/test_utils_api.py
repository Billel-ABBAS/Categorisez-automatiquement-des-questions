import mlflow
import mlflow.sklearn
import os
import sys

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



########################## Teste de la fonction clean_html ############################
import pytest
from bs4 import BeautifulSoup
import re
from utils.utils_exploratoire import clean_html

@pytest.mark.parametrize("input_html, expected_output", [
    ("<p>Visit our site <a href='http://example.com'>here</a></p>", "Visit our site "),  # Test de suppression des URLs
    ("<div><style>body {color: red;}</style>Text with style</div>", "Text with style"),  # Test de suppression des balises style
    ("<script>alert('Hello');</script>Warning text", "Warning text"),  # Test de suppression des balises script
    ("<div>Hello <code>Code snippet</code> world</div>", "Hello   world")  # Test de suppression des balises code, mais en les conservant
    
])
def test_clean_html(input_html, expected_output):
    result = clean_html(input_html)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"



########################## Teste de la fonction process_clean_text ############################

import pytest
from utils.utils_exploratoire import process_clean_text, stop_words  



@pytest.mark.parametrize("doc, rejoin, expected_result", [
    ("Learn Python, Java, and C++ today!", True, "learn python java c++ today"),  
    ("Error in script.js at line 10", False, ["error", "scriptjs", "line"]),  
    ("Updating node.js and npm packages", True, "nodejs npm package"),  
    ("Use #include <iostream> in C++", True, "use include iostream c++"),  
 
])
def test_process_clean_text(doc, rejoin, expected_result):
    result = process_clean_text(doc, rejoin)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
    



########################## Teste de la fonction jaccard  ############################


import pytest
import numpy as np
from sklearn.metrics import jaccard_score
from utils.utils_supervised import jaccard


# teste 1: Vérification du calcul du score de Jaccard pour différents scénarios standards.

@pytest.mark.parametrize("y_true, y_pred, expected_score", [
    (np.array([[1, 0, 1], [1, 1, 0]]), np.array([[1, 0, 0], [1, 0, 0]]), 0.5),  # Cas simple
    (np.array([[0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 0], [0, 0, 1]]), 1.0),        # Correspondance parfaite
    (np.array([[1, 1, 1], [0, 0, 0]]), np.array([[0, 0, 0], [1, 1, 1]]), 0.0),        # Aucune correspondance
])
def test_jaccard(y_true, y_pred, expected_score):
    result = jaccard(y_true, y_pred)
    np.testing.assert_almost_equal(result, expected_score, decimal=7, err_msg=f"Expected {expected_score}, but got {result}")


# teste 2: Tester des cas aux extrémités pour vérifier que la fonction gère correctement les situations comme des entrées complètement vides ou des entrées contenant uniquement des zéros ou des uns.

def test_jaccard_edge_cases():
    # Cas où les deux tableaux sont vides
    y_true = np.array([[]])
    y_pred = np.array([[]])
    result = jaccard(y_true, y_pred)
    assert np.isnan(result), "Expected NaN for empty arrays"

    # Cas où les deux tableaux sont remplis de zéros
    y_true = np.zeros((3, 3))
    y_pred = np.zeros((3, 3))
    result = jaccard(y_true, y_pred)
    assert result == 1.0, "Expected Jaccard score of 1.0 when both arrays are zero"

    # Cas où les deux tableaux sont remplis de uns
    y_true = np.ones((3, 3))
    y_pred = np.ones((3, 3))
    result = jaccard(y_true, y_pred)
    assert result == 1.0, "Expected Jaccard score of 1.0 when both arrays are one"
    
    

########################## Teste de la fonction predict_tags ############################

import pytest
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils_supervised import predict_tags

# Récupérer les identifiants AWS depuis les variables d'environnement configurées dans Github
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configurer MLflow pour utiliser un chemin compatible avec WSL
mlflow.set_tracking_uri("http://ec2-44-204-37-245.compute-1.amazonaws.com:5000/")

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
    logged_vectorizer = 'runs:/c6d4a460a6d4425f8a9e9db7d374c0c0/vectorizer_supervised'
    tfidf_vectorizer = load_mlflow_model(logged_vectorizer, 'sklearn')

    # Charger le MultiLabelBinarizer depuis MLflow
    logged_mlb = 'runs:/272a9f59470c400a989d26a7555929ce/mlb'
    mlb = load_mlflow_model(logged_mlb, 'sklearn')

    # Charger le modèle XGBoost en tant que Sklearn Model depuis MLflow
    logged_model = 'runs:/a4ead1bae4f9424bb0ec41236e5bd50d/XGBoost'
    model = load_mlflow_model(logged_model, 'sklearn')

    return model, tfidf_vectorizer, mlb

# Exemple de texte à prédire
@pytest.fixture
def example_text():
    return '''
    'java socket binding port bind socket client port code connection choose port exception case exception connect love pointer'
    '''


# exemple de teste 1:
def test_predict_tags(setup_mlflow_models, example_text):
    model, tfidf_vectorizer, mlb = setup_mlflow_models
    
    # Appeler la fonction pour prédire les tags
    predicted_tags = predict_tags(example_text, model, tfidf_vectorizer, mlb, top_n=5)
    
    # Exemple de tags attendus (cela dépend du modèle entraîné et des données)
    expected_tags = ['sql', 'exception', 'java']
    
    # Vérifier que les tags prédits contiennent au moins certains des tags attendus
    assert all(tag in predicted_tags for tag in expected_tags), \
        f"Expected some of the tags {expected_tags} to be in {predicted_tags}"


# exemple de teste 2:
def test_predict_tags_edge_cases_with_tech_tags(setup_mlflow_models):
    model, tfidf_vectorizer, mlb = setup_mlflow_models
    
       # Test 1: Texte très long (répétition de mots clés)
    long_text = " ".join(["python float int string conversion c++ java javascript"] * 1000)
    predicted_tags_long = predict_tags(long_text, model, tfidf_vectorizer, mlb, top_n=5)
    assert len(predicted_tags_long) > 0, "Expected some tags for long text, but got none."
    # Vérifie que des tags informatiques sont bien présents
    tech_tags = ['python', 'c++', 'java', 'javascript']
    assert any(tag in predicted_tags_long for tag in tech_tags), \
        f"Expected at least one of the tech tags {tech_tags} in {predicted_tags_long}"

        # Test 2: Texte contenant uniquement des tags informatiques
    tech_text = "python java c++ javascript"
    predicted_tags_tech = predict_tags(tech_text, model, tfidf_vectorizer, mlb, top_n=5)
    expected_tech_tags =  [ 'java', 'javascript', 'python']
    assert all(tag in predicted_tags_tech for tag in expected_tech_tags), \
        f"Expected all tech tags {expected_tech_tags} to be in {predicted_tags_tech}"
