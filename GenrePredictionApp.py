import streamlit as st
import joblib
import numpy as np
import re
import nltk
from gensim.models import Word2Vec


# Ensure NLTK resources are available
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


# Loading Trained Models
# Load trained Word2Vec model
w2v_model = Word2Vec.load("models/word2vec_model.model")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load the saved scaler
standard_scaler = joblib.load('models/scaler.pkl')

# Load the saved LightGBM model
LightGBM_model = joblib.load('models/lightgbm_model.pkl')

# Load the trained models
model_paths = {
    "LightGBM": 'models/lightgbm_model.pkl',
    "XGradient Boost": 'models/xgboost_model.pkl',
    "Random Forest": 'models/random_forest_model.pkl'
}

models = {name: joblib.load(path) for name, path in model_paths.items()}


# ~~~~~~~~~~~~~~~~ Cleaning Functions ~~~~~~~~~~~~~~~~ #
# Load custom stopwords from stopwords.txt
def load_custom_stopwords(filepath):
    with open(filepath, 'r') as file:
        stopwords_set = set(line.strip().lower() for line in file)
    return stopwords_set

# Path to stopwords.txt
stopwords_path = 'data/stopwords/english'
custom_stopwords = load_custom_stopwords(stopwords_path)

# Data cleaning function
def normalize(text):
    text = text.lower()
    # remove singing noises
    text = re.sub(r'\bm+\b', ' ', text)
    text = re.sub(r'\bo+h+\b', ' ', text)
    text = re.sub(r'\ba+h+\b', ' ', text)
    text = re.sub(r'\bh+m+\b', ' ', text)
    text = re.sub(r'\by+o+\b', ' ', text)
    text = re.sub(r'\bo+y+\b', ' ', text)
    text = re.sub(r'\bg+o+\b', ' ', text)
    text = re.sub(r'\bu+m+\b', ' ', text)

    # normalize unnecessary repeated characters
    text = re.sub(r'\bo+n+\b', ' on ', text)
    text = re.sub(r'\bn+o+\b', ' no ', text)
    text = re.sub(r'\bn+o+w+\b', ' now ', text)
    text = re.sub(r'\by+o+u+\b', ' you ', text)
    text = re.sub(r'\by+e+a+h+\b', ' yeah ', text)
    text = re.sub(r'\bb+a+b+y+\b', ' baby ', text)
    text = re.sub(r'\bw+a+n+t+\b', ' want ', text)
    text = re.sub(r'\bt+r+u+s+t+\b', ' trust ', text)

    # correct some misspelled words
    text = re.sub(r'\bl+u+v+\b', ' love ', text)
    text = re.sub(r'\bl+o+v+\b', ' love ', text)

    return text


def clean_text(text):
    text = re.sub(r'\[.*?\]', ' ', text)  # Remove text between brackets
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Removes everything except letters and spaces

    # Tokenize and remove stopwords
    words = text.strip().split()
    stop_words = set(word.lower() for word in custom_stopwords)
    filtered_words = [word for word in words if word not in stop_words]

    # Remove words of length 1 or 2
    filtered_words = [word for word in filtered_words if len(word) > 2]

    # Lemmatize words
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words) # Join the filtered words back into a single string

# Tokenization: split lyrics into lists of words
def tokenize_text(text):
    tokens = text.split()
    return tokens


# ~~~~~~~~~~~~~~~~ Feature Extraction Function ~~~~~~~~~~~~~~~~ #
# Word2Vec features
def compute_weighted_w2v_vector(tokens, model, tfidf_weights, vector_size):
    """
    Compute a weighted Word2Vec vector for tokens using TF-IDF weights.

    Args:
        tokens (list): List of word tokens.
        model (Word2Vec): Trained Word2Vec model.
        tfidf_weights (dict): Mapping of words to their TF-IDF weights.
        vector_size (int): Dimension of the Word2Vec vectors.

    Returns:
        np.ndarray: Weighted sum of Word2Vec vectors for the tokens.
    """
    word_vectors = []
    for word in tokens:
        if word in model.wv and word in tfidf_weights:
            weight = tfidf_weights.get(word, 1.0)  # Default to 1.0 if word not in TF-IDF
            word_vectors.append(weight * model.wv[word])
    if not word_vectors:
        return np.zeros(vector_size)  # Return zero vector if no valid words
    return np.sum(word_vectors, axis=0)  # Compute the weighted sum

# custom features
def count_baby_occurrences(text):
    return text.split().count('baby')

def get_average_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0

def get_count_words(text):
    return len(text.split())

def get_custom_features(text):
    baby_occurrences = count_baby_occurrences(text)
    avg_word_length = get_average_word_length(text)
    count_words = get_count_words(text)

    return [baby_occurrences, avg_word_length, count_words]

# Input Processing Function
def preprocess_input(lyrics):
    """Preprocess user input lyrics and return the feature vector."""
    lyrics = normalize(lyrics)
    lyrics = clean_text(lyrics)
    tokens = tokenize_text(lyrics)

    # get Word2Vec features
    vocab = tfidf_vectorizer.get_feature_names_out()
    tfidf_weights = dict(zip(vocab, tfidf_vectorizer.idf_))  # Map words to their TF-IDF weights
    w2v_vector = compute_weighted_w2v_vector(tokens, w2v_model, tfidf_weights, w2v_model.vector_size)

    # get custom features
    custom_features = get_custom_features(lyrics)

    # combine word2vec features with custom ones
    combined_features = np.hstack((custom_features, w2v_vector))

    return standard_scaler.transform([combined_features])  # Scale the vector


# ~~~~~~~~~~~~~~~~ Streamlit web app ~~~~~~~~~~~~~~~~ #
# command to run the app --> streamlit run GenrePredictionApp.py
st.title("Lyrics Genre Predictor")
st.write("Enter the lyrics of a song to predict its genre.")

# User input
user_input = st.text_area("Song Lyrics", height=250)

# Model selection
selected_model_name = st.selectbox("Choose a model for prediction:", list(models.keys()))
selected_model = models[selected_model_name]

if st.button("Predict Genre"):
    if not user_input.strip():
        st.warning("Please enter some lyrics!")
    else:
        # Preprocess input
        input_vector = preprocess_input(user_input)

        # Make prediction
        prediction = selected_model.predict(input_vector)

        # Map prediction to genre
        genre_mapping = {0: 'Country', 1: 'Pop', 2: 'Rap', 3: 'RnB', 4: 'Rock'}  # Adjust based on your label encoding
        predicted_genre = genre_mapping.get(prediction[0], "Unknown")

        # Display result
        st.success(f"The predicted genre is: **{predicted_genre}**")
