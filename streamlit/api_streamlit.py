import streamlit as st
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import sys

# Récupérer les identifiants AWS depuis les variables d'environnement configurées dans Streamlit Cloud
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import utils_exploratoire as func_exploratoire
from utils import utils_supervised as func_supervised

# import utils as func

# Configurer MLflow pour utiliser un chemin compatible avec WSL
mlflow.set_tracking_uri("http://ec2-44-204-37-245.compute-1.amazonaws.com:5000/")

# Interface utilisateur avec Streamlit
st.title('Prédiction de mots-clés avec le Meilleur Modèle')

# Afficher un message indiquant que le modèle est chargé
st.write("Chargement des modèles et artefacts depuis MLflow...")

# Fonction pour charger un modèle MLflow avec mise en cache
@st.cache_resource
def load_mlflow_model(model_uri, model_type='sklearn'):
  try:
      if model_type == 'sklearn':
          return mlflow.sklearn.load_model(model_uri)
      else:
          raise ValueError("Type de modèle non supporté")
  except Exception as e:
      st.error(f"Erreur lors du chargement du modèle {model_uri}: {str(e)}")
      return None

# Charger les artefacts depuis MLflow

# Charger le TfidfVectorizer depuis MLflow
logged_vectorizer = 'runs:/c6d4a460a6d4425f8a9e9db7d374c0c0/vectorizer_supervised'
vectorizer_supervised = load_mlflow_model(logged_vectorizer, 'sklearn')
#   if vectorizer_supervised is not None:
#       st.success("Vectorizer supervisé chargé avec succès!")

# Charger le MultiLabelBinarizer depuis MLflow
logged_mlb = 'runs:/272a9f59470c400a989d26a7555929ce/mlb'
mlb = load_mlflow_model(logged_mlb, 'sklearn')
#   if mlb is not None:
#       st.success("MultiLabelBinarizer chargé avec succès!")

# Charger le modèle XGBoost en tant que PyFuncModel depuis MLflow
logged_model = 'runs:/a4ead1bae4f9424bb0ec41236e5bd50d/XGBoost'
best_model = load_mlflow_model(logged_model, 'sklearn')
#   if best_model is not None:
#       st.success("Meilleur modèle XGBoost chargé avec succès!")



# Entrée de texte par l'utilisateur
user_input = st.text_area("Entrez le texte pour la prédiction des mots-clés:")

if st.button('Prédire'):
  if user_input:
      # Nettoyage et tokenisation du texte d'entrée
      cleaned_html_input = func_exploratoire.clean_html(user_input)
    #   cleaned_html_input = func.clean_html(user_input)
      cleaned_input = func_exploratoire.process_clean_text(cleaned_html_input)
    #   cleaned_input = func.process_clean_text(cleaned_html_input)
      
      # Prédiction des mots-clés avec le modèle supervisé
      predicted_tags = func_supervised.predict_tags(cleaned_input, best_model, vectorizer_supervised, mlb)
    #   predicted_tags = func.predict_tags(cleaned_input, best_model, vectorizer_supervised, mlb)
      
      # Afficher les résultats
      st.subheader('Texte nettoyé:')
      st.write(cleaned_input)
      
      st.subheader('Mots-clés prédits:')
      st.write(predicted_tags)
  else:
      st.error("Veuillez entrer du texte pour la prédiction.")


