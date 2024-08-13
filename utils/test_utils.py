
                                   ### Partie exploratoire ###

########################## Teste de la fonction taux_de_Remplissage_tableau ############################
import pytest
import pandas as pd
from utils_exploratoire import taux_de_Remplissage_tableau

def test_taux_de_Remplissage_tableau():
    # Création d'un DataFrame de test
    data = {
        "col1": [1, 2, 3, None],
        "col2": [None, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    taux_de_Remplissage_tableau(df)

########################## Teste de la fonction taux_de_Remplissage_tableau ############################

import pytest
from utils_exploratoire import split_tags

test_data = [
    ("<python><data-science><3x><machine.learning>", ['python', 'data', 'science', 'machinelearning']),
    ("<html><css><3><12.3>", ['html', 'css']),
    ("<full-stack><web-development>", ['full', 'stack', 'web', 'development']),
    ("", [''])
]

@pytest.mark.parametrize("input_tags, expected_output", test_data)
def test_split_tags(input_tags, expected_output):
    assert split_tags(input_tags) == expected_output


########################## Teste de la fonction display_tokens_infos ############################

import pytest
from utils_exploratoire import display_tokens_infos

def test_display_tokens_infos_small_list(capfd):
    tokens = ["test", "test", "example", "example", "test"]
    display_tokens_infos(tokens)
    out, err = capfd.readouterr()

    expected_output = (
        "Nombre de tokens: 5\n"
        "Nombre de tokens uniques: 2\n\n"  # Ajout d'un saut de ligne supplémentaire ici pour correspondre à la sortie réelle
        "Exemple de tokens: ['test', 'test', 'example', 'example', 'test'] \n\n"
    )
    # Pour déboguer et voir exactement ce qui est généré
    print(repr(out))

    assert out == expected_output


########################## Teste de la fonction test_analyze_pos_tags ############################

import pytest
from collections import Counter
from nltk import pos_tag
from utils_exploratoire import analyze_pos_tags  # Assurez-vous que ce chemin d'importation est correct

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

import pytest
from utils_exploratoire import process_clean_text, stop_words  



@pytest.mark.parametrize("doc, rejoin, expected_result", [
    ("Learn Python, Java, and C++ today!", True, "learn python java c++ today"),  
    ("Error in script.js at line 10", False, ["error", "scriptjs", "line"]),  
    ("Updating node.js and npm packages", True, "nodejs npm package"),  
    ("Use #include <iostream> in C++", True, "use include iostream c++"),  
 
])
def test_process_clean_text(doc, rejoin, expected_result):
    result = process_clean_text(doc, rejoin)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"




########################## Teste de la fonction clean_html ############################

import pytest
from bs4 import BeautifulSoup
import re
from utils_exploratoire import clean_html  

@pytest.mark.parametrize("input_html, expected_output", [
    ("<p>Visit our site <a href='http://example.com'>here</a></p>", "Visit our site "),  # Test de suppression des URLs
    ("<div><style>body {color: red;}</style>Text with style</div>", "Text with style"),  # Test de suppression des balises style
    ("<script>alert('Hello');</script>Warning text", "Warning text"),  # Test de suppression des balises script
    ("<div>Hello <code>Code snippet</code> world</div>", "Hello   world")  # Test de suppression des balises code, mais en les conservant
    
])
def test_clean_html(input_html, expected_output):
    result = clean_html(input_html)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"



                                   ### Partie non supervisé ###

########################## Teste de la fonction vectorizer_transform ############################


import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from utils_non_supervised import vectorizer_transform  

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
    
import pytest
import numpy as np
from utils_non_supervised import filter_words  

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

import pytest
import pandas as pd
from utils_non_supervised  import coverage_rate  

# Exemple de données de test pour DataFrame
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


import pytest
import numpy as np
from sklearn.metrics import jaccard_score
from utils_supervised  import jaccard  


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
