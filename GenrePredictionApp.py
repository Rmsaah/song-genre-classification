import streamlit as st
import joblib
import numpy as np
import re
import nltk
from gensim.models import KeyedVectors

# Has to be the first Streamlit call in the script.
st.set_page_config(
    page_title="Song Genre Classifier",
    page_icon="🎵",
    layout="centered",
)

# The five genres the models were trained on, in LabelEncoder order
# (country -> 0, pop -> 1, rap -> 2, rb -> 3, rock -> 4; see Main.ipynb).
GENRES = ["Country", "Pop", "Rap", "RnB", "Rock"]
GENRE_ICONS = {"Country": "🤠", "Pop": "✨", "Rap": "🎤", "RnB": "🎹", "Rock": "🎸"}

# Accuracy on the held-out 60,000 song test set, from Main.ipynb.
MODEL_ACCURACY = {"LightGBM": 0.6765, "XGradient Boost": 0.6759}


# Ensure NLTK resources are available
@st.cache_resource
def download_nltk_data():
    """WordNetLemmatizer needs the wordnet corpus, which nltk does not bundle."""
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


download_nltk_data()


# Loading Trained Models
@st.cache_resource
def load_models():
    """Load once and keep in memory, instead of re-reading on every rerun."""
    # word2vec.kv keeps only the 8000 vectors the TF-IDF vocabulary can reach,
    # so it is 7.8 MB instead of 190 MB. See README for how to rebuild it.
    w2v_model = KeyedVectors.load("models/word2vec.kv")
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    standard_scaler = joblib.load('models/scaler.pkl')

    # Random Forest is deliberately left out. The trained model is 720 MB,
    # which is over GitHub's 100 MB per-file limit, so it cannot be deployed.
    # LightGBM and XGBoost both scored higher anyway (68% vs 66%).
    model_paths = {
        "LightGBM": 'models/lightgbm_model.pkl',
        "XGradient Boost": 'models/xgboost_model.pkl'
    }
    models = {name: joblib.load(path) for name, path in model_paths.items()}

    return w2v_model, tfidf_vectorizer, standard_scaler, models


w2v_model, tfidf_vectorizer, standard_scaler, models = load_models()


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
    text = re.sub(r'\bu+h+\b', ' ', text)
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
        model (KeyedVectors): Trained Word2Vec vectors.
        tfidf_weights (dict): Mapping of words to their TF-IDF weights.
        vector_size (int): Dimension of the Word2Vec vectors.

    Returns:
        np.ndarray: Weighted sum of Word2Vec vectors for the tokens.
    """
    word_vectors = []
    for word in tokens:
        if word in model and word in tfidf_weights:
            weight = tfidf_weights.get(word, 1.0)  # Default to 1.0 if word not in TF-IDF
            word_vectors.append(weight * model[word])
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
    """Preprocess user input lyrics and return the feature vector.

    Also returns how many tokens the models actually recognized, so the app can
    say when there is nothing to go on instead of dressing up a majority-class
    guess as a prediction. That count is zero for anything not in English.
    """
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
    # Order has to match training (Main.ipynb): Word2Vec first, then handcrafted.
    # Passing them the other way round silently halves accuracy, since the
    # scaler and classifiers expect the 250 Word2Vec columns first.
    combined_features = np.hstack((w2v_vector, custom_features))

    # Same condition compute_weighted_w2v_vector uses to keep a word.
    matched = sum(1 for t in tokens if t in w2v_model and t in tfidf_weights)

    return standard_scaler.transform([combined_features]), matched  # Scale the vector


# ~~~~~~~~~~~~~~~~ Streamlit web app ~~~~~~~~~~~~~~~~ #
# command to run the app --> streamlit run GenrePredictionApp.py

# Colors are neutral greys with alpha rather than fixed values, so the app
# reads the same whether the visitor is on Streamlit's light or dark theme.
st.markdown(
    """
    <style>
    .genre-chips { margin: 0.1rem 0 0.6rem 0; }
    .genre-chip {
        display: inline-block;
        padding: 0.28rem 0.8rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        background: rgba(128, 128, 128, 0.12);
        font-size: 0.9rem;
        white-space: nowrap;
    }
    .result-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-left: 4px solid #FF4B4B;
        border-radius: 0.6rem;
        background: rgba(128, 128, 128, 0.08);
        padding: 1rem 1.25rem;
        margin-bottom: 1.4rem;
    }
    .result-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        opacity: 0.6;
    }
    .result-genre { font-size: 2rem; font-weight: 700; line-height: 1.3; }
    .result-conf { font-size: 0.9rem; opacity: 0.75; }
    .prob-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.45rem 0; }
    .prob-label { flex: 0 0 7rem; font-size: 0.92rem; }
    .prob-track {
        flex: 1 1 auto;
        min-width: 2rem;
        height: 0.55rem;
        border-radius: 999px;
        background: rgba(128, 128, 128, 0.18);
        overflow: hidden;
    }
    .prob-fill { height: 100%; border-radius: 999px; background: #FF4B4B; }
    .prob-value {
        flex: 0 0 3.4rem;
        text-align: right;
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        opacity: 0.8;
    }
    .prob-row.runner-up .prob-fill { opacity: 0.32; }
    .prob-row.runner-up .prob-label,
    .prob-row.runner-up .prob-value { opacity: 0.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app guesses a song's genre from its **lyrics alone** - no audio, "
        "no artist, no release year. It was trained on 300,000 English songs "
        "scraped from Genius."
    )

    with st.expander("How it works"):
        st.markdown(
            "1. **Clean** - strip section headers like `[Chorus]`, punctuation, "
            "stopwords and singing noises (*ooh*, *yeah*), then lemmatize.\n"
            "2. **Vectorize** - turn the words into a 250-dimension Word2Vec "
            "vector, weighted by TF-IDF so distinctive words count for more.\n"
            "3. **Add features** - word count, average word length, and how "
            "often the word *baby* shows up.\n"
            "4. **Classify** - feed all 253 features to a gradient-boosted "
            "tree model."
        )

    with st.expander("How accurate is it?"):
        st.markdown(
            "About **68%** overall, but that average hides a lot. The training "
            "data was mostly Pop and Rap, so the model is far better at those "
            "than at the rarer genres:\n\n"
            "| Genre | Correctly found |\n"
            "| --- | --- |\n"
            "| 🎤 Rap | 87% |\n"
            "| ✨ Pop | 86% |\n"
            "| 🎸 Rock | 22% |\n"
            "| 🤠 Country | 10% |\n"
            "| 🎹 RnB | 7% |\n\n"
            "*Recall on a held-out 60,000 song test set, LightGBM. XGBoost is "
            "within a couple of points.*"
        )

st.title("🎵 Song Genre Classifier")
st.markdown(
    "Paste the lyrics of a song and find out which genre it most likely belongs "
    "to - predicted from the words alone, with no audio."
)

st.markdown("**Genres it can pick from**")
st.markdown(
    '<div class="genre-chips">'
    + "".join(f'<span class="genre-chip">{GENRE_ICONS[g]} {g}</span>' for g in GENRES)
    + "</div>",
    unsafe_allow_html=True,
)

st.info(
    "**English lyrics only.** The models were trained on English songs, and "
    "cleaning strips every non-Latin character - lyrics in another language are "
    "reduced to nothing, so the prediction is meaningless.",
    icon="🌐",
)

# User input
user_input = st.text_area(
    "Song lyrics",
    height=260,
    placeholder=(
        "Paste the lyrics here...\n\n"
        "Section headers like [Verse 1] and [Chorus] are fine, they get removed "
        "automatically."
    ),
)

# Model selection
selected_model_name = st.selectbox(
    "Model",
    list(models.keys()),
    help="Both are gradient-boosted tree models, and they score within 0.1% of each other.",
)
selected_model = models[selected_model_name]
st.caption(f"Test-set accuracy: **{MODEL_ACCURACY[selected_model_name]:.1%}**")

if st.button("Predict genre", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some lyrics first.")
    else:
        # Preprocess input
        input_vector, matched_words = preprocess_input(user_input)

        if matched_words == 0:
            # Every word was stripped or is out of vocabulary. The model would
            # still return the majority class, which would be pure noise.
            st.error(
                "None of those words are in the model's vocabulary, so there is "
                "nothing to classify. This almost always means the lyrics are "
                "not in English.",
                icon="🚫",
            )
        else:
            if matched_words < 5:
                st.warning(
                    f"Only {matched_words} word(s) were recognized, so this is "
                    "little more than a guess. Try a full set of lyrics.",
                    icon="⚠️",
                )

            # Make prediction. Column i of predict_proba is class i, which maps
            # onto GENRES by the LabelEncoder order noted at the top of the file.
            probabilities = selected_model.predict_proba(input_vector)[0]
            ranking = np.argsort(probabilities)[::-1]
            top_genre = GENRES[int(ranking[0])]
            confidence = float(probabilities[ranking[0]])

            # Display result
            st.markdown(
                '<div class="result-card">'
                '<div class="result-eyebrow">Predicted genre</div>'
                f'<div class="result-genre">{GENRE_ICONS[top_genre]} {top_genre}</div>'
                f'<div class="result-conf">{confidence:.0%} confidence</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            # All five scores, so a close call is visible rather than hidden.
            st.markdown("**Full breakdown**")
            rows = []
            for rank, index in enumerate(ranking):
                genre = GENRES[int(index)]
                probability = float(probabilities[index])
                css_class = "prob-row" if rank == 0 else "prob-row runner-up"
                rows.append(
                    f'<div class="{css_class}">'
                    f'<div class="prob-label">{GENRE_ICONS[genre]} {genre}</div>'
                    '<div class="prob-track">'
                    f'<div class="prob-fill" style="width:{probability * 100:.1f}%"></div>'
                    "</div>"
                    f'<div class="prob-value">{probability:.1%}</div>'
                    "</div>"
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

            if confidence < 0.4:
                st.caption(
                    "The model is not confident here, so treat the top genre as "
                    "a weak preference rather than an answer."
                )
