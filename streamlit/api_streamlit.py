import streamlit as st

# Importations des fonctions personnalisées
import init_streamlit as func



# Interface utilisateur avec Streamlit
st.title('Prédiction de mots-clés avec LDA et NMF')

# Entrée de texte par l'utilisateur
user_input = st.text_area("Entrez le texte pour la prédiction des mots-clés:")

if st.button('Prédire'):
    if user_input:
        try:
            # Nettoyage et tokenisation du texte d'entrée
            cleaned_html_input = func.clean_html(user_input)
            cleaned_input = func.process_clean_text(cleaned_html_input)
            
            # Charger les modèles et les données
            lda_model, nmf_model, vectorizer, vectorizer_tags, train_topics_lda, train_topics_nmf, tags_train = func.load_models_and_data() 
            
            # Prédiction des mots-clés avec LDA et NMF
            predicted_keywords_lda, predicted_semi_supervised_keywords_lda, predicted_keywords_nmf, predicted_semi_supervised_keywords_nmf = func.predict_keywords(
                cleaned_input, lda_model, nmf_model, vectorizer, train_topics_lda, train_topics_nmf, tags_train, vectorizer_tags
            )
            
            st.subheader('Texte nettoyé:')
            st.write(cleaned_input)
            
            st.subheader('Mots-clés prédits avec le modèle LDA:')
            st.write(predicted_keywords_lda)
            
            st.subheader('Mots-clés semi-supervisés prédits avec le modèle LDA:')
            st.write(predicted_semi_supervised_keywords_lda)
            
            st.subheader('Mots-clés prédits avec le modèle NMF:')
            st.write(predicted_keywords_nmf)
            
            st.subheader('Mots-clés semi-supervisés prédits avec le modèle NMF:')
            st.write(predicted_semi_supervised_keywords_nmf)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
    else:
        st.error("Veuillez entrer du texte pour la prédiction.")

