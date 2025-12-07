import pickle
import pandas as pd
from feature_extractor import extract_url_features

class PhishingModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.model_name = None
        
    def load_models(self):
        """Charge le modèle, le scaler et les features sauvegardés"""
        try:
            self.model = pickle.load(open('phishing_model_optimized.pkl', 'rb'))
            self.scaler = pickle.load(open('scaler_optimized.pkl', 'rb'))
            self.features = pickle.load(open('features_optimized.pkl', 'rb'))
            self.model_name = "Random Forest"  # À adapter selon le modèle sauvegardé
            print("✅ Modèles chargés avec succès")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
            return False
    
    def predict_url(self, url):
        """Prédit si une URL est du phishing"""
        if not self.model or not self.scaler or not self.features:
            print("❌ Modèles non chargés")
            return None
        
        try:
            print(f"\n🔍 Analyse de: {url}")
            
            # Extraire features
            url_features = extract_url_features(url)
            if not url_features:
                return None
            
            # Préparer les données
            features_df = pd.DataFrame([url_features])
            features_df = features_df[self.features]
            
            # Prédiction
            if self.model_name in ['SVM', 'Regression Logistique', 'Naive Bayes']:
                features_scaled = self.scaler.transform(features_df)
                prediction = self.model.predict(features_scaled)[0]
                probability = self.model.predict_proba(features_scaled)[0][1]
            else:
                prediction = self.model.predict(features_df)[0]
                probability = self.model.predict_proba(features_df)[0][1]
            
            # Niveau de confiance
            if probability > 0.8 or probability < 0.2:
                confidence = "Élevée"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Moyenne"
            else:
                confidence = "Faible"
            
            result = {
                'url': url,
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': confidence,
                'model_used': self.model_name,
                'features': {k: v for k, v in url_features.items() if k in self.features}
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return None

# Instance globale
phishing_model = PhishingModel()

def load_phishing_model():
    """Charge le modèle de phishing"""
    return phishing_model.load_models()

def predict_phishing(url):
    """Prédit si une URL est du phishing"""
    return phishing_model.predict_url(url)

if __name__ == '__main__':
    # Test du chargement et prédiction
    if load_phishing_model():
        test_url = "https://www.google.com"
        result = predict_phishing(test_url)
        if result:
            print(f"Résultat: {result}")