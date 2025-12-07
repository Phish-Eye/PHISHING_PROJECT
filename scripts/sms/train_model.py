# ================================
# train_model.py - Email Phishing Detection
# ================================

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from nltk.corpus import stopwords
import nltk
import pickle
import os
# ================================
# Download NLTK data
# ================================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ================================
# Load Dataset
# ================================
path = os.path.join('..', '..', 'datasets', 'phishing_email.csv')
df = pd.read_csv(path)
    
print(df.head())

# ================================
# Text Cleaning & Stopwords Removal (Optimized)
# ================================
stop_words = set(stopwords.words('english'))

def fast_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()                  # plus rapide que word_tokenize
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned_text"] = df["text_combined"].astype(str).apply(fast_clean)

# ================================
# Vectorization (TF-IDF with bigrams)
# ================================
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

# ================================
# Split Dataset
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Train Logistic Regression Model
# ================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# ================================
# Evaluate Model
# ================================
y_pred_log = log_model.predict(X_test)
print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_log))
ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test)

# ================================
# Save Model & Vectorizer
# ================================
with open("./vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("./log_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

print("✅ Model and vectorizer saved successfully.")