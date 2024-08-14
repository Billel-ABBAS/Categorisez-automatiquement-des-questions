## Introduction

Ce projet a été réalisé dans le cadre de la formation d'ingénieur machine learning proposée par OpenClassrooms. L'objectif principal était de développer une API de suggestion de tags pour les utilisateurs de Stack Overflow, en utilisant des techniques de traitement automatique du langage naturel (NLP). Ce repository se concentre sur l'analyse exploratoire des données, ainsi que sur l'entraînement de modèles supervisés et non supervisés pour répondre à ce besoin.

Les principales étapes du projet comprenaient :

- Le filtrage des données issues de l'[API StackExchange Explorer](https://data.stackexchange.com/stackoverflow/query/new).
- Le prétraitement des documents textuels.
- La comparaison d'approches supervisées (Logistic Regression, SGD Classifier, SVM, XGBoost, Random Forest, LightGBM, et AdaBoost) et non supervisées (LDA, NMF, semi-supervisé LDA et semi-supervisé NMF) pour la prédiction des tags.
- Le test unitaire de chaque fonction avec **pytest** en local et dans GitHub.
- Le développement et la mise en production de l'API dans Streamlit Cloud, qui utilise des modèles stockés dans MLflow.

## Contenu du Repository

Le repository contient les éléments suivants :

1. **Abbas_Billel_1_notebook_requete_API_072024.ipynb** : Ce notebook est consacré aux requêtes API pour extraire les données depuis StackExchange Explorer.

2. **Abbas_Billel_2_notebook_exploration_072024.ipynb** : Ce notebook est dédié à l'analyse exploratoire des données afin de comprendre la structure des données et les préparer pour la modélisation.

3. **Abbas_Billel_3_notebook_approche_non_supervisée_072024.ipynb** : Ce notebook couvre les différentes étapes du prétraitement des données textuelles et l'entraînement des modèles non supervisés.

4. **Abbas_Billel_4_notebook_approche_supervisée_072024.ipynb** : Ce notebook traite de l'entraînement des modèles supervisés.

Le repository contient 3 dossiers :

- **`mlflow`** : Ce dossier contient un module Python pour l'entraînement des modèles supervisés, en utilisant **MLflow** pour suivre les expériences et stocker les résultats. Le meilleur modèle est automatiquement sauvegardé dans un bucket AWS. On stocke dans ce même bucket AWS le modèle **MLB** (MultiLabelBinarizer) ainsi que le **TfidfVectorizer**. Ces modèles seront utilisés par l'API de prédiction déployée sur Streamlit.

- **`streamlit`** : Ce dossier contient le module Python `api-streamlit.py` et le fichier `requirements.txt` nécessaires pour lancer l'API sur Streamlit Cloud. L'API utilise les modèles sauvegardés pour fournir des suggestions de tags en fonction des questions posées par les utilisateurs de Stack Overflow. Le dossier contient également `test_utils_api.py`, un script de test qui, à chaque `git push`, déclenche un workflow **pytest** dans GitHub grâce au fichier `.github/workflows/main.yml`.

- **`utils`** : Ce dossier regroupe les modules Python contenant les fonctions nécessaires pour l'analyse exploratoire des données, ainsi que pour les approches supervisée et non supervisée. Plus précisément, il inclut :
  - `utils_exploratoire.py` : Ce module contient les fonctions pour l'analyse exploratoire des données.
  - `utils_non_supervised.py` : Ce module regroupe les fonctions utilisées pour les approches non supervisées.
  - `utils_supervised.py` : Ce module contient les fonctions dédiées aux méthodes supervisées.
  - `test_utils.py` : Ce fichier permet de tester localement les fonctions définies dans ces modules à l'aide de **pytest**.

## Données

Les données utilisées dans ce projet proviennent de l'API StackExchange Explorer, qui permet d'extraire des questions, réponses, et tags associés de la plateforme Stack Overflow. Ces données ont été filtrées, nettoyées, et prétraitées avant d'être utilisées pour l'entraînement des modèles.

## Modèles Entraînés

Les modèles suivants ont été comparés dans ce projet :

- **Approches supervisées** :
  - Logistic Regression
  - SGD Classifier
  - Support Vector Machine (SVM)
  - XGBoost
  - Random Forest
  - LightGBM
  - AdaBoost

- **Approches non supervisées** :
  - Latent Dirichlet Allocation (LDA)
  - Non-Negative Matrix Factorization (NMF)
  - Modèle semi-supervisé LDA
  - Modèle semi-supervisé NMF

## Choix du Modèle pour l'API

Le modèle XGBoost s'est avéré être le plus performant parmi les modèles supervisés, avec un score de Jaccard moyen de 0.622 et un taux de couverture de 0.27. Ce taux de couverture était significativement meilleur que celui obtenu avec les approches non supervisées (0.1 pour LDA et NMF, et < 0.0001 pour les modèles semi-supervisés). Grâce à ce taux de couverture supérieur, le modèle XGBoost supervisé a été choisi comme le meilleur modèle pour être implémenté dans l'API Streamlit.


## Déploiement de l'API

L'API de suggestion de tags, basée sur le modèle XGBoost, est déployée sur Streamlit Cloud. Vous pouvez accéder à l'API via le lien suivant :

[Accéder à l'API de suggestion de tags sur Streamlit Cloud](https://projetsopc-nzffdgnvmwzl8kbnjaf7lq.streamlit.app/)

## Suivi des Modèles avec MLflow

Les expérimentations et le suivi des modèles ont été effectués à l'aide de **MLflow**. Vous pouvez accéder au tableau de bord MLflow pour visualiser les différentes expériences et les modèles entraînés via le lien suivant :

[Accéder au tableau de bord MLflow](http://ec2-44-204-37-245.compute-1.amazonaws.com:5000/#/experiments/473337626577195962?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

