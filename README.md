# Song Genre Classification

* [Overview](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#overview)
* [How it Works](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#how-it-works)
* [Setup Instructions](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#setup-instructions)
* [Models](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#models)
* [Streamlit Application](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#streamlit-application)
* [Future Work](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#future-work)

## Overview
The main goal of this project is to leverage NLP and Machine Learning to classify song lyrics (English) by genre, without needing any audio data. The [dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) was collected from Genius, and it has around five million records.

## How it Works
**1. Data Collection & Cleaning**
   * Normalized some unnecessarily repeated characters.
   * Removed stopwords, symbols and punctuation.

**2. Feature Extraction**
   * Lemmatize & Tokenzie words.
   * Used Word2Vec for feature extraction, with TF-IDF adding weight to words.

**3. Model Training**
   * Experimented with different algorithms.
   * Picked out the best preforming ones (LightGBM, Random Forest, XGBoost).

**4. Evaluation**
   * Compared using accuracy, precision, recall and F1-score.
   * LightGBM      --> Accuracy: 68%
   * Random Forest --> Accuracy: 66%
   * XGBoost       --> Accuracy: 68%

**5. Deployment**
   * Integrated the trained models into a Streamlit web application for easy interaction.
   * The deployed app serves **LightGBM** and **XGBoost**. Random Forest is still trained and
     evaluated in the notebook, but it is left out of the app: the saved model is 720 MB, which
     is over GitHub's 100 MB per-file limit, and it was the weakest of the three anyway.

## Setup Instructions
**1. Clone the Repository**
```
git clone https://github.com/rmsaah/song-genre-classification.git
cd song-genre-classification
```

**2. Install Dependencies**
```
pip install -r requirements.txt
```

**3. Get the Dataset (only needed to re-run the notebook)**

The datasets are not in this repository, since `song_lyrics.csv` alone is 9 GB. Download it from
the [Kaggle link](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
above and place it in `data/`. The app itself does not need it and only reads `data/stopwords/english`.

## Models
`models/` holds only what the app needs at prediction time, about 14 MB in total:

| File | Size | Purpose |
| --- | --- | --- |
| `word2vec.kv` | 7.8 MB | Word2Vec vectors, trimmed to the TF-IDF vocabulary |
| `lightgbm_model.pkl` | 3.4 MB | LightGBM classifier |
| `xgboost_model.pkl` | 2.3 MB | XGBoost classifier |
| `tfidf_vectorizer.pkl` | 0.3 MB | TF-IDF weights used to weight the Word2Vec aggregation |
| `scaler.pkl` | 6.6 KB | StandardScaler fitted on the 253 training features |

Two training artifacts are deliberately kept out of git (see `.gitignore`):

* **`random_forest_model.pkl` (720 MB)** — over GitHub's 100 MB per-file limit, so it cannot be
  deployed. It also scored lowest of the three models.
* **`word2vec_model.model` and its two `.npy` files (190 MB)** — the full model, 98,298 words.

The app can only ever use a vector when a word is in **both** the Word2Vec vocabulary and the
TF-IDF vocabulary, and TF-IDF was fitted with `max_features=8000`. So `word2vec.kv` keeps just
those 8000 vectors: identical predictions, 24x smaller. Rebuild it after retraining with:

```python
import joblib
from gensim.models import Word2Vec, KeyedVectors

full = Word2Vec.load("models/word2vec_model.model")
vocab = joblib.load("models/tfidf_vectorizer.pkl").get_feature_names_out()

keep = [w for w in vocab if w in full.wv]
slim = KeyedVectors(vector_size=full.vector_size)
slim.add_vectors(keep, [full.wv[w] for w in keep])
slim.save("models/word2vec.kv", sep_limit=512 * 1024 ** 2)
```

## Streamlit Application
To run the Streamlit application, navigate to the directory containing the GenrePredictionApp.py and run the following command:
```
streamlit run GenrePredictionApp.py
```

## Future Work
The current approaches use classical Machine Learning for classification. I plan on exploring Deep Learning algorithms to enhance the models performance in the future.
