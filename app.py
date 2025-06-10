import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import pickle
import traceback
import base64
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AgeRate SubClassifier",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AgeRate SubClassifier")
st.markdown("""
* **G**: General Audiences (Semua Umur)
* **PG**: Parental Guidance Suggested (Bimbingan Orang Tua)
* **PG-13**: Parents Strongly Cautioned (Di bawah 13 tahun perlu pendampingan orang tua)
* **R**: Restricted (Di bawah 17 tahun perlu pendampingan orang tua)
""")

RF_MODEL_PATH = "randomforest_model/random_forest_pipeline.pkl"

POSSIBLE_PATHS = [
    {
        "NB_PATH": "naivebayes_model/complete_pipeline.pkl",
        "NB_CLASSIFIER_PATH": "naivebayes_model/naive_bayes_classifier.pkl",
        "NB_VECTORIZER_PATH": "naivebayes_model/vectorizer.pkl",
        "NB_FEATURE_NAMES_PATH": "naivebayes_model/feature_names.npy",
        "NB_MODEL_INFO_PATH": "naivebayes_model/model_info.json",        
    }
]

def find_valid_nb_paths():
    for path_set in POSSIBLE_PATHS:
        if os.path.exists(path_set["NB_PATH"]):
            return path_set
    
    return POSSIBLE_PATHS[0]

PATHS = find_valid_nb_paths()
NB_PATH = PATHS["NB_PATH"]
NB_CLASSIFIER_PATH = PATHS["NB_CLASSIFIER_PATH"]
NB_VECTORIZER_PATH = PATHS["NB_VECTORIZER_PATH"]
NB_FEATURE_NAMES_PATH = PATHS["NB_FEATURE_NAMES_PATH"]
NB_MODEL_INFO_PATH = PATHS["NB_MODEL_INFO_PATH"]

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.warning(f"⚠️ Gagal mengunduh NLTK resources: {str(e)}")
        return False

nltk_available = download_nltk_resources()

def clean_subtitle(text):
    text = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_subtitle(text)
    text = text.lower()
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        tokens = text.split()
    try:
        nltk_stopwords = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in nltk_stopwords]
    except Exception as e:
        pass

    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except Exception as e:
        pass
    
    tokens = [word for word in tokens if len(word) >= 3]
    
    processed_text = ' '.join(tokens)
    
    return processed_text

    
@st.cache_resource
def load_models():
    models = {}
    
    tried_paths = []
    
    try:
        vectorizer = joblib.load(NB_VECTORIZER_PATH)
        classifier = joblib.load(NB_CLASSIFIER_PATH)
        
        nb_model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        models['naive_bayes'] = nb_model
    except Exception as e:
        tried_paths.append(f"Kombinasi vectorizer & classifier: {str(e)}")
        st.warning(f"⚠️ Model Naive Bayes tidak dapat dimuat")

    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        models['random_forest'] = rf_model

    except Exception as e:
        st.warning(f"⚠️ Model Random Forest tidak dapat dimuat: {e}")

    return models


def get_important_features_nb(text, model):
    # Preprocess teks
    processed_text = preprocess_text(text)
    
    try:
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']
        
        X_vec = vectorizer.transform([processed_text])

        try:
            feature_names = vectorizer.get_feature_names_out()
        except:
            try:
                feature_names = vectorizer.get_feature_names()
            except:
                feature_names = [f"feature_{i}" for i in range(X_vec.shape[1])]

        nonzero_features = X_vec.nonzero()[1]

        feature_importance = {}
        
        for i, category in enumerate(classifier.classes_):
            importance_scores = []
            
            for feature_idx in nonzero_features:
                feature_value = X_vec[0, feature_idx]
   
                feature_log_prob = classifier.feature_log_prob_[i, feature_idx]

                contribution = feature_value * feature_log_prob
                
                importance_scores.append((feature_names[feature_idx], contribution))

            importance_scores.sort(key=lambda x: x[1], reverse=True)

            feature_importance[category] = importance_scores
        
        return feature_importance
    except Exception as e:
        feature_importance = {}
        for category in ['G', 'PG', 'PG-13', 'R']:
            words = processed_text.split()  # Ambil semua kata
            importance_scores = [(word, 0.1) for word in words]
            feature_importance[category] = importance_scores
        return feature_importance

def get_important_features_rf(text, model):
    processed_text = preprocess_text(text)
    
    try:
        vectorizer = model.named_steps['vectorizer']

        X_vec = vectorizer.transform([processed_text])

        try:
            feature_names = vectorizer.get_feature_names_out()
        except:
            try:
                feature_names = vectorizer.get_feature_names()
            except:
                feature_names = [f"feature_{i}" for i in range(X_vec.shape[1])]
        
        nonzero_features = X_vec.nonzero()[1]
        nonzero_values = X_vec[0, nonzero_features].toarray()[0]

        detected_words = [(feature_names[idx], nonzero_values[i]) for i, idx in enumerate(nonzero_features)]
  
        detected_words.sort(key=lambda x: x[1], reverse=True)
        
        return detected_words  
    except Exception as e:
        words = processed_text.split()
        dummy_words = [(word, 0.5) for word in words]
        return dummy_words

def get_prediction_proba(text, model):
    processed_text = preprocess_text(text)
    
    try:
        probas = model.predict_proba([processed_text])[0]

        classes = model.classes_

        result = {}
        for i, cls in enumerate(classes):
            result[cls] = probas[i]
        
        return result
    except Exception as e:
        return {'G': 0.25, 'PG': 0.25, 'PG-13': 0.25, 'R': 0.25}

def get_prediction(text, model):
    processed_text = preprocess_text(text)
    try:
        prediction = model.predict([processed_text])[0]
        probas = get_prediction_proba(text, model)
        
        return prediction, probas
    except Exception as e:
        prediction = "PG-13" 
        probas = {'G': 0.25, 'PG': 0.25, 'PG-13': 0.25, 'R': 0.25}
        return prediction, probas

def plot_probability(probas, title):
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = list(probas.keys())
    values = list(probas.values())
    
    if all(v == 0 for v in values):
        ax.text(0.5, 0.5, "Tidak ada data probabilitas", 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Rating Film')
    
    return fig

def format_prediction_result(prediction, probas):
    rating_descriptions = {
        'G': 'General Audiences (Semua Umur)',
        'PG': 'Parental Guidance Suggested (Bimbingan Orang Tua)',
        'PG-13': 'Parents Strongly Cautioned (Di bawah 13 tahun perlu pendampingan orang tua)',
        'R': 'Restricted (Di bawah 17 tahun perlu pendampingan orang tua)'
    }
    
    rating_colors = {
        'G': '#2ecc71',  # Hijau
        'PG': '#3498db',  # Biru
        'PG-13': '#f39c12',  # Oranye
        'R': '#e74c3c'  # Merah
    }
    
    if prediction not in rating_descriptions:
        result_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h3 style="color: #e74c3c;">Error: {prediction}</h3>
            <p>Tidak dapat memprediksi rating film.</p>
        </div>
        """
        return result_html
    

    result_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <h3 style="color: {rating_colors[prediction]};">{prediction}: {rating_descriptions[prediction]}</h3>
        <p>Probabilitas per kategori:</p>
        <ul>
    """
    
    for rating, prob in probas.items():
        result_html += f'<li><b>{rating}</b>: {prob:.4f}</li>'
    
    result_html += """
        </ul>
    </div>
    """
    
    return result_html

def display_important_words(importance_data, model_type):
    if model_type == "naive_bayes":
        all_words = []
        for category, words in importance_data.items():
            for word, contrib in words:
                all_words.append((word, abs(contrib)))
    
        word_dict = {}
        for word, value in all_words:
            if word not in word_dict or value > word_dict[word]:
                word_dict[word] = value

        unique_words = [(word, value) for word, value in word_dict.items()]
        unique_words.sort(key=lambda x: x[1], reverse=True)
     
        st.write("**Semua Kata yang Terdeteksi dalam Teks:**")
 
        df = pd.DataFrame(unique_words, columns=["Kata", "Nilai Kontribusi"])
        
        def color_value(val):
            return f'background-color: rgba(52, 152, 219, {min(val*2, 0.8)})'
        
        st.dataframe(df.style.applymap(color_value, subset=['Nilai Kontribusi']))
    else:

        try:
            vectorizer = models[model_type].named_steps['vectorizer']

            processed_text = preprocess_text(text_input)

            X_vec = vectorizer.transform([processed_text])

            try:
                feature_names = vectorizer.get_feature_names_out()
            except:
                try:
                    feature_names = vectorizer.get_feature_names()
                except:
                    feature_names = [f"feature_{i}" for i in range(X_vec.shape[1])]

            nonzero_features = X_vec.nonzero()[1]
            nonzero_values = X_vec[0, nonzero_features].toarray()[0]
 
            detected_words = [(feature_names[idx], nonzero_values[i]) for i, idx in enumerate(nonzero_features)]
        
            detected_words.sort(key=lambda x: x[1], reverse=True)
            
            importance_data = detected_words
        except Exception as e:

            importance_data = importance_data
            
        st.write("**Semua Kata yang Terdeteksi dalam Teks:**")

        df = pd.DataFrame(importance_data, columns=["Kata", "Nilai TF-IDF"])
        
        def color_tfidf(val):
            return f'background-color: rgba(52, 152, 219, {min(val*2, 0.8)})'
        
        st.dataframe(df.style.applymap(color_tfidf, subset=['Nilai TF-IDF']))

models = load_models()

def main():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tentang Aplikasi")
    st.sidebar.info("""
    Aplikasi ini menggunakan model machine learning Naive Bayes dan Random Forest untuk memprediksi rating film (G, PG, PG-13, R) berdasarkan teks subtitle.
    """)
    
    available_models = list(models.keys())

    if len(available_models) > 1:
        if 'naive_bayes' in models and 'random_forest' in models:
            model_options = ["Naive Bayes", "Random Forest", "Keduanya"]
        else:
            model_options = []
            if 'naive_bayes' in models:
                model_options.append("Naive Bayes")
            if 'random_forest' in models:
                model_options.append("Random Forest")
            if 'dummy' in models:
                model_options.append("Dummy Model (Demo)")
        
        model_option = st.sidebar.radio(
            "Pilih Model Prediksi:",
            model_options
        )
    elif len(available_models) == 1:
        if 'naive_bayes' in models:
            model_option = "Naive Bayes"
        elif 'random_forest' in models:
            model_option = "Random Forest"
        elif 'dummy' in models:
            model_option = "Dummy Model (Demo)"
        
        st.sidebar.info(f"Hanya model {model_option} yang tersedia")
    else:
        st.error("Tidak ada model yang tersedia. Pastikan file model berada di lokasi yang benar.")
        return
    
    uploaded_file = st.file_uploader("📂Unggah file subtitle (.srt)", type=["srt", "txt"])
    
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read().decode("utf-8", errors="ignore")

            if not file_contents or file_contents.isspace():
                st.error("File yang diunggah kosong.")
                text_input = ""
                predict_button = False
            else:
                with st.expander("Preview File", expanded=False):
                    st.text(file_contents[:1000] + ("..." if len(file_contents) > 1000 else ""))

                text_input = file_contents
                predict_button = st.button("Prediksi Rating Film")
        except Exception as e:
            st.error(f"Error saat membaca file: {str(e)}")
            text_input = ""
            predict_button = False
    else:
        text_input = ""
        predict_button = False

    if predict_button and text_input:
        progress_bar = st.progress(0)

        def handle_prediction(model_name, model_key):
            if model_key in models:
                try:
                    with st.spinner(f"Memprediksi dengan {model_name}..."):
                        progress_bar.progress(33)
    
                        processed_text = preprocess_text(text_input)
                       
                        try:
                            prediction = models[model_key].predict([processed_text])[0]
                            
                            try:
                                probas = models[model_key].predict_proba([processed_text])[0]
                                classes = models[model_key].classes_
                                
                                probas_dict = {}
                                for i, cls in enumerate(classes):
                                    probas_dict[cls] = probas[i]
                            except Exception as e:
                                probas_dict = {'G': 0.25, 'PG': 0.25, 'PG-13': 0.25, 'R': 0.25}
                                if prediction in probas_dict:
                                    probas_dict[prediction] = 0.7  
                        except Exception as e:
                            prediction = "Tidak dapat memprediksi"
                            probas_dict = {'G': 0, 'PG': 0, 'PG-13': 0, 'R': 0}

                        progress_bar.progress(66)

                        st.markdown(format_prediction_result(prediction, probas_dict), unsafe_allow_html=True)

                        st.subheader("Grafik Probabilitas")
                        fig = plot_probability(probas_dict, f"Probabilitas Rating Film - {model_name}")
                        st.pyplot(fig)

                        st.subheader("Analisis Kata Penting")
                        try:
                            if model_key == 'naive_bayes':
                                important_features = get_important_features_nb(text_input, models[model_key])
                                display_important_words(important_features, "naive_bayes")
                            elif model_key == 'random_forest':
                                detected_words = get_important_features_rf(text_input, models[model_key])
                                display_important_words(detected_words, "random_forest")
                            else:  
                                st.info("Analisis kata penting tidak tersedia untuk Dummy Model")
                        except Exception as e:
                            st.error(f"❌ Error saat menganalisis kata penting: {str(e)}")
                        
                        return prediction, probas_dict
                except Exception as e:
                    st.error(f"❌ Error umum saat memprediksi dengan {model_name}: {str(e)}")
                    return None, None
            else:
                st.warning(f"Model {model_name} tidak tersedia.")
                return None, None
        
 
        if model_option == "Keduanya" and 'naive_bayes' in models and 'random_forest' in models:
            tab1, tab2, tab3 = st.tabs(["Naive Bayes", "Random Forest", "Perbandingan"])
        
            nb_prediction, nb_probas = None, None
            rf_prediction, rf_probas = None, None
        
            try:
                progress_bar.progress(25)
                processed_text = preprocess_text(text_input)
                nb_prediction = models['naive_bayes'].predict([processed_text])[0]
                nb_probas_array = models['naive_bayes'].predict_proba([processed_text])[0]
                nb_classes = models['naive_bayes'].classes_

                nb_probas = {}
                for i, cls in enumerate(nb_classes):
                    nb_probas[cls] = nb_probas_array[i]
                    
                progress_bar.progress(50)

                rf_prediction = models['random_forest'].predict([processed_text])[0]
                rf_probas_array = models['random_forest'].predict_proba([processed_text])[0]
                rf_classes = models['random_forest'].classes_
                

                rf_probas = {}
                for i, cls in enumerate(rf_classes):
                    rf_probas[cls] = rf_probas_array[i]
                    
                progress_bar.progress(75)
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

            with tab1:
                if nb_prediction is not None:
                    st.markdown(format_prediction_result(nb_prediction, nb_probas), unsafe_allow_html=True)

                    st.subheader("Grafik Probabilitas")
                    fig_nb = plot_probability(nb_probas, "Probabilitas Rating Film - Naive Bayes")
                    st.pyplot(fig_nb)

                    st.subheader("Analisis Kata Penting")
                    try:
                        important_features = get_important_features_nb(text_input, models['naive_bayes'])
                        display_important_words(important_features, "naive_bayes")
                    except Exception as e:
                        st.error(f"❌ Error saat menganalisis kata penting: {str(e)}")
  
                else:
                    st.error("❌ Gagal melakukan prediksi dengan model Naive Bayes.")

            with tab2:
                if rf_prediction is not None:
                    st.markdown(format_prediction_result(rf_prediction, rf_probas), unsafe_allow_html=True)
                    st.subheader("Grafik Probabilitas")
                    fig_rf = plot_probability(rf_probas, "Probabilitas Rating Film - Random Forest")
                    st.pyplot(fig_rf)
                    st.subheader("Analisis Kata Penting")
                    try:
                        detected_words = get_important_features_rf(text_input, models['random_forest'])
                        display_important_words(detected_words, "random_forest")
                    except Exception as e:
                        st.error(f"❌ Error saat menganalisis kata penting: {str(e)}")
                else:
                    st.error("❌ Gagal melakukan prediksi dengan model Random Forest.")
            
            with tab3:
                if nb_prediction and rf_prediction:
                    st.subheader("Perbandingan Hasil Kedua Model")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Naive Bayes:** {nb_prediction}")
                        fig_nb = plot_probability(nb_probas, "Naive Bayes")
                        st.pyplot(fig_nb)
                        
                    with col2:
                        st.markdown(f"**Random Forest:** {rf_prediction}")
                        fig_rf = plot_probability(rf_probas, "Random Forest")
                        st.pyplot(fig_rf)
                    

                    if nb_prediction == rf_prediction:
                        st.success(f"✅ Kedua model memprediksi rating yang sama: {nb_prediction}")
                        st.markdown("Ini menunjukkan tingkat kepercayaan yang tinggi terhadap hasil prediksi.")
                    else:
                        st.warning(f"⚠️ Model memberikan prediksi yang berbeda: NB={nb_prediction}, RF={rf_prediction}")
                        st.markdown("Ini menunjukkan bahwa konten subtitle mungkin memiliki karakteristik yang berada di perbatasan kategori rating.")

                        nb_confidence = max(nb_probas.values())
                        rf_confidence = max(rf_probas.values())
                        
                        if nb_confidence > rf_confidence:
                            st.info(f"💡 Model Naive Bayes memiliki tingkat kepercayaan lebih tinggi ({nb_confidence:.2f} vs {rf_confidence:.2f}).")
                        else:
                            st.info(f"💡 Model Random Forest memiliki tingkat kepercayaan lebih tinggi ({rf_confidence:.2f} vs {nb_confidence:.2f}).")
                else:
                    st.warning("⚠️ Salah satu atau kedua model gagal memberikan prediksi yang valid.")
                    if not nb_prediction:
                        st.error("❌ Prediksi Naive Bayes gagal.")
                    if not rf_prediction:
                        st.error("❌ Prediksi Random Forest gagal.")

            progress_bar.progress(100)
        
        elif model_option == "Naive Bayes" or (model_option == "Dummy Model (Demo)" and 'dummy' in models):
            model_key = 'naive_bayes' if model_option == "Naive Bayes" else 'dummy'
            model_name = "Naive Bayes" if model_option == "Naive Bayes" else "Dummy Model"
            nb_prediction, nb_probas = handle_prediction(model_name, model_key)
            
            progress_bar.progress(100)
        
        elif model_option == "Random Forest":
            rf_prediction, rf_probas = handle_prediction("Random Forest", 'random_forest')           

            progress_bar.progress(100)
            

if __name__ == "__main__":
    main()