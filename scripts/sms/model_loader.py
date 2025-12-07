# ================================
# model_loader.py
# Charge le modèle + function predict_email()
# ================================

import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------
# Charger modèle et vectorizer
# ------------------------------
with open("scripts/NLP/log_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scripts/NLP/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------------
# Nettoyage du texte
# ------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def remove_stopwords(text):
    words = word_tokenize(text)
    return ' '.join([w for w in words if w not in stop_words])

# ------------------------------
# Fonction prédiction
# ------------------------------
def predict_email(text):
    text = clean_text(text)
    text = remove_stopwords(text)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return prediction