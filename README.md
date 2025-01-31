# Song Genre Classification

* [Overview](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#overview)
* [How it Works](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#how-it-works)
* [Setup Instructions](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#setup-instructions)
* [Streamlit Application](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#streamlit-application)
* [Future Work](https://github.com/Rmsaah/song-genre-classification?tab=readme-ov-file#future-work)

## Overview
The main goal of this project is to leverage NLP and Machine Learning to classify song lyrics by genre, without needing any audio data. The [dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) was collected from Genius, and it has around five million records.

## How it Works
**1. Data Collection & Cleaning**
   * Kept only necessary columns (lyrics, tag).
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

## Setup Instructions
1. Clone the Repository
```
git clone https://github.com/rmsaah/song-genre-classification.git
cd song-genre-classification
```

2. Install Dependencies
```
pip install -r requirements.txt
```

## Streamlit Application
To run the Streamlit application, navigate to the directory containing the GenrePredictionApp.py and run the following command:
```
streamlit run GenrePredictionApp.py
```

## Future Work
The current approaches use classical Machine Learning for classification. I plan on exploring Deep Learning algorithms to enhance the models performance in the future.
