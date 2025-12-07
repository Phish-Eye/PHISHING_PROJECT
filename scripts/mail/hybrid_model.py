
import pickle
import numpy as np
import re
import string
from nltk.corpus import stopwords
import nltk

# Télécharger NLTK
nltk.download('stopwords', quiet=True)

# ================================
# FONCTION DE NETTOYAGE GLOBALE
# ================================
def fast_clean(text):
    """Nettoie le texte comme dans le modèle contenu"""
    if not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ================================
# FONCTION DE PRÉDICTION HYBRIDE
# ================================
def hybrid_predict(content, url, content_model, content_vectorizer, 
                  url_model, url_scaler, url_features):
    """Prédiction hybride simple (moyenne des probabilités)"""
    
    # 1. Prédiction contenu
    cleaned_content = fast_clean(content)
    content_vectorized = content_vectorizer.transform([cleaned_content])
    prob_content = content_model.predict_proba(content_vectorized)[0][1]
    
    # 2. Prédiction URL
    url_feats = {
        'length_url': len(url),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_slash': url.count('/'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_eq': url.count('='),
        'has_https': 1 if url.startswith('https://') else 0,
        'has_http': 1 if url.startswith('http://') else 0,
        'has_www': 1 if 'www.' in url.lower() else 0,
        'suspicious_tld': 1 if any(tld in url.lower() for tld in ['.tk', '.ml', '.ga', '.cf', '.gq']) else 0
    }
    
    feat_vector = []
    for feat in url_features:
        feat_vector.append(url_feats.get(feat, 0))
    
    feat_vector = np.array(feat_vector).reshape(1, -1)
    feat_scaled = url_scaler.transform(feat_vector)
    prob_url = url_model.predict_proba(feat_scaled)[0][1]
    
    # 3. Moyenne pondérée
    final_prob = (0.6 * prob_content + 0.4 * prob_url)
    
    # 4. Logique de décision améliorée
    if prob_content > 0.8 or prob_url > 0.8:
        final_prob = max(prob_content, prob_url)
    elif abs(prob_content - prob_url) > 0.3:
        final_prob = prob_content if prob_content > prob_url else prob_url
    
    return {
        'phishing_probability': float(final_prob),
        'is_phishing': final_prob > 0.5,
        'content_prob': float(prob_content),
        'url_prob': float(prob_url),
        'agreement': (prob_content > 0.5) == (prob_url > 0.5),
        'confidence_level': 'HIGH' if final_prob > 0.8 else 'MEDIUM' if final_prob > 0.6 else 'LOW',
        'details': {
            'content_decision': 'PHISHING' if prob_content > 0.5 else 'LEGITIMATE',
            'url_decision': 'PHISHING' if prob_url > 0.5 else 'LEGITIMATE',
            'final_decision': 'PHISHING' if final_prob > 0.5 else 'LEGITIMATE'
        }
    }

# ================================
# CHARGEMENT ET SAUVEGARDE
# ================================
def create_and_save_hybrid():
    """Charge les modèles et sauvegarde les composants hybrides SANS la fonction !"""
    print("🔧 Création du modèle hybride...")
    
    try:
        print("📥 Chargement du modèle contenu...")
        with open('../sms/log_model.pkl', 'rb') as f:
            content_model = pickle.load(f)
        
        with open('../sms/vectorizer.pkl', 'rb') as f:
            content_vectorizer = pickle.load(f)
        
        print("📥 Chargement du modèle URL...")
        with open('../url/phishing_model_optimized.pkl', 'rb') as f:
            url_model = pickle.load(f)
        
        with open('../url/scaler_optimized.pkl', 'rb') as f:
            url_scaler = pickle.load(f)
        
        with open('../url/features_optimized.pkl', 'rb') as f:
            url_features = pickle.load(f)
        
        print("✅ Modèles chargés avec succès!")
        
        # ⚠️ ICI : On NE sauvegarde PAS la fonction hybrid_predict
        hybrid_components = {
            'content_model': content_model,
            'content_vectorizer': content_vectorizer,
            'url_model': url_model,
            'url_scaler': url_scaler,
            'url_features': url_features
        }
        
        with open('hybrid_model_full.pkl', 'wb') as f:
            pickle.dump(hybrid_components, f)
        
        print("✅ Modèle hybride sauvegardé: hybrid_model_full.pkl")
        
        return hybrid_components
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

# ================================
# CHARGEMENT
# ================================
def load_hybrid_model():
    """Charge les composants hybrides"""
    try:
        with open('hybrid_model_full.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("⚠️ Modèle hybride non trouvé → Création...")
        return create_and_save_hybrid()

# ================================
# FONCTION POUR STREAMLIT
# ================================
def predict_with_hybrid(content, url, hybrid_components=None):
    """Utilise le modèle hybride sans jamais charger la fonction depuis pickle"""
    if hybrid_components is None:
        hybrid_components = load_hybrid_model()
    
    if hybrid_components is None:
        return {
            'error': 'Modèle hybride non disponible',
            'phishing_probability': 0.5,
            'is_phishing': False
        }
    
    # ⚠️ Appel direct → plus besoin de hybrid_components['predict_function']
    return hybrid_predict(
        content=content,
        url=url,
        content_model=hybrid_components['content_model'],
        content_vectorizer=hybrid_components['content_vectorizer'],
        url_model=hybrid_components['url_model'],
        url_scaler=hybrid_components['url_scaler'],
        url_features=hybrid_components['url_features']
    )

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 CRÉATION DU MODÈLE HYBRIDE PHISHING")
    print("=" * 60)
    
    hybrid_model = create_and_save_hybrid()
    
    if hybrid_model:
        print("\n🎉 Modèle hybride créé avec succès!")
