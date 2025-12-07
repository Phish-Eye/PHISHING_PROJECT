import streamlit as st
import re
import string
import pickle
from nltk.corpus import stopwords
import nltk
import os

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
def setup_page():
    """Configuration de la page"""
    st.set_page_config(
        page_title="Détecteur de Spam/Phishing",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé amélioré
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            padding-top: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .phishing-alert {
            background: linear-gradient(135deg, #ffcccc 0%, #ff9999 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #ff0000;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(255, 0, 0, 0.15);
        }
        .safe-alert {
            background: linear-gradient(135deg, #ccffcc 0%, #99ff99 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #00cc00;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0, 255, 0, 0.15);
        }
        .info-box {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 10px 0;
        }
        .keyword-tag {
            display: inline-block;
            background-color: #ffebee;
            color: #c62828;
            padding: 4px 10px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.85rem;
            border: 1px solid #ffcdd2;
        }
        .feature-card {
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 4px solid #6c757d;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# FONCTIONS DE PRÉTRAITEMENT
# ============================================
def initialize_nltk():
    """Initialise NLTK si nécessaire"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def fast_clean(text):
    """Nettoie le texte comme pendant l'entraînement"""
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

# ============================================
# CHARGEMENT DES MODÈLES
# ============================================
@st.cache_resource
def load_models():
    """Charge le modèle et le vectorizer"""
    try:
        with open("./vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("./log_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model, True
    except FileNotFoundError:
        st.error("❌ Fichiers .pkl non trouvés dans le dossier courant!")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None, None, False

# ============================================
# SIDEBAR
# ============================================
def display_sidebar():
    """Affiche la sidebar avec informations"""
    with st.sidebar:
        st.header("ℹ️ Informations")
        
        st.markdown("""
        **📊 À propos du modèle:**
        - Algorithme: Régression Logistique
        - Vectorisation: TF-IDF avec bigrammes
        - Précision: >95% sur emails anglais
        - Entraîné sur: Dataset d'emails phishing
        """)
        
        st.markdown("---")
        st.header("🎯 Conseils d'utilisation")
        
        st.markdown("""
        **Pour de meilleurs résultats:**
        - Utilisez des textes en **anglais**
        - Le modèle est optimisé pour les **emails**
        - Les SMS français peuvent donner des résultats moins précis
        - Vérifiez toujours manuellement en cas de doute
        """)
        
        st.markdown("---")
        st.header("📋 Exemples rapides")
        
        # Boutons d'exemples
        examples = {
            "Phishing typique": "URGENT: Your bank account needs verification. Click: http://secure-bank-login.com",
            "Promotion suspecte": "CONGRATULATIONS! You won $5000! Claim now: http://win-prize.com",
            "Email légitime": "Dear customer, your order #12345 has been shipped. Thank you for shopping!",
            "Message normal": "Hello team, meeting tomorrow at 10 AM in conference room B."
        }
        
        for name, text in examples.items():
            if st.button(f"📝 {name}", use_container_width=True, key=f"ex_{name}"):
                st.session_state.email_text = text
                st.rerun()
        
        st.markdown("---")
        st.caption("🔒 Système de détection IA")

# ============================================
# LISTE DES MOTS-CLÉS
# ============================================
def get_phishing_keywords():
    """Retourne la liste des mots-clés de phishing"""
    return [
        "account","access","action","alert","attention","authenticate","authentication","bank",
        "billing","block","browser","buy","cancel","certificate","click","confirm","confirmation",
        "contact","credential","credit","danger","deactivate","delivery","download","email",
        "enforce","ensure","error","expires","fail","failure","finance","form","important","immediately",
        "information","identity","illegal","insecure","invoice","issue","login","logon","mail","member",
        "message","money","notification","password","payment","personal","phish","press","priority",
        "problem","protect","protection","purchase","re-authenticate","re-enter","recovery","refund",
        "risk","security","secure","service","signin","suspend","suspicious","update","urgent","verify",
        "verification","validate","virus","warning","web","website","wire"
    ]

# ============================================
# CONTENU PRINCIPAL
# ============================================
def display_main_content(vectorizer, model):
    """Affiche le contenu principal de l'application"""
    # En-tête
    st.markdown('<h1 class="main-header">📧 Détecteur de Spam/Phishing</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Analyse intelligente de messages texte avec IA</h2>', unsafe_allow_html=True)
    
    # Section d'analyse
    st.markdown("---")
    st.subheader("🔍 Analyse de message")
    
    # Zone de texte
    if 'email_text' not in st.session_state:
        st.session_state.email_text = ""
    
    email_text = st.text_area(
        "Entrez le texte à analyser:",
        value=st.session_state.email_text,
        height=200,
        placeholder="Collez ici votre email, SMS ou message texte...\n\nExemple: URGENT: Your account needs verification. Click here: http://secure-login.com",
        help="Le modèle est optimisé pour les textes en anglais."
    )
    
    # Boutons d'action
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_btn = st.button(
            "🚀 Lancer l'analyse", 
            type="primary",
            use_container_width=True,
            help="Cliquez pour analyser le texte avec l'IA"
        )
    
    with col2:
        if st.button("🧹 Effacer", use_container_width=True):
            st.session_state.email_text = ""
            st.rerun()
    

    # Analyse
    if analyze_btn and email_text.strip():
        perform_analysis(email_text, vectorizer, model)
    elif analyze_btn and not email_text.strip():
        st.warning("⚠️ Veuillez entrer un texte à analyser.")
    
    # Section éducative
    display_educational_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
        "🔍 Système de détection IA • Modèle: Régression Logistique • Version 1.0"
        "</div>", 
        unsafe_allow_html=True
    )

# ============================================
# ANALYSE PRINCIPALE
# ============================================
def perform_analysis(email_text, vectorizer, model):
    """Effectue l'analyse du texte"""
    with st.spinner("🔍 Analyse en cours... L'IA examine votre message"):
        # Prétraitement
        cleaned = fast_clean(email_text)
        
        # Prédiction
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        # Détection mots-clés
        phishing_keywords = get_phishing_keywords()
        words = cleaned.split()
        detected_keywords = [w for w in words if w in phishing_keywords]
    
    # Affichage des résultats
    st.markdown("---")
    
    if prediction == 1:
        display_phishing_result(confidence, detected_keywords, email_text, cleaned, probabilities)
    else:
        display_safe_result(confidence, detected_keywords, email_text, cleaned, probabilities)

# ============================================
# AFFICHAGE RÉSULTATS PHISHING
# ============================================
def display_phishing_result(confidence, detected_keywords, original_text, cleaned_text, probabilities):
    """Affiche les résultats pour un message phishing"""
    st.markdown(f"""
    <div class="phishing-alert">
        <h2>🚨 PHISHING/SPAM DÉTECTÉ</h2>
        <h3>Niveau de risque: ÉLEVÉ</h3>
        <p><strong>Confiance du modèle:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilité phishing", f"{probabilities[1]:.1%}")
    
    with col2:
        st.metric("Mots-clés détectés", len(detected_keywords))
    
    with col3:
        risk_level = "CRITIQUE" if confidence > 0.9 else "ÉLEVÉ" if confidence > 0.7 else "MODÉRÉ"
        st.metric("Niveau d'alerte", risk_level)
    
    # Recommandations
    st.warning("""
    **⚠️ RECOMMANDATIONS DE SÉCURITÉ:**
    - **Ne cliquez pas** sur les liens suspects
    - **Ne répondez pas** au message
    - **Ne partagez pas** d'informations personnelles
    - **Signalez** le message comme spam
    - **Contactez** l'organisation via son site officiel
    """)
    
    # Détails techniques
    display_technical_details(detected_keywords, original_text, cleaned_text, probabilities, is_phishing=True)

# ============================================
# AFFICHAGE RÉSULTATS SÉCURISÉS
# ============================================
def display_safe_result(confidence, detected_keywords, original_text, cleaned_text, probabilities):
    """Affiche les résultats pour un message légitime"""
    st.markdown(f"""
    <div class="safe-alert">
        <h2>✅ MESSAGE LÉGITIME</h2>
        <h3>Niveau de risque: FAIBLE</h3>
        <p><strong>Confiance du modèle:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilité légitime", f"{probabilities[0]:.1%}")
    
    with col2:
        st.metric("Mots-clés détectés", len(detected_keywords))
    
    with col3:
        safety_level = "TRÈS SÛR" if confidence > 0.9 else "SÛR" if confidence > 0.7 else "MODÉRÉ"
        st.metric("Niveau de sécurité", safety_level)
    
    st.info("""
    **✅ CE MESSAGE SEMBLE SÉCURISÉ:**
    - Ton conversationnel normal
    - Pas de sentiment d'urgence artificiel
    - Pas de demandes suspectes
    - Mots-clés appropriés au contexte
    """)
    
    # Détails techniques
    display_technical_details(detected_keywords, original_text, cleaned_text, probabilities, is_phishing=False)

# ============================================
# DÉTAILS TECHNIQUES
# ============================================
def display_technical_details(detected_keywords, original_text, cleaned_text, probabilities, is_phishing):
    """Affiche les détails techniques de l'analyse"""
    with st.expander("📊 DÉTAILS TECHNIQUES DE L'ANALYSE", expanded=True):
        tab1, tab2, tab3 = st.tabs(["📝 Texte", "🔑 Mots-clés", "📈 Statistiques"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Texte original:**")
                st.code(original_text[:500] + "..." if len(original_text) > 500 else original_text, language="text")
            
            with col2:
                st.write("**Texte nettoyé:**")
                st.code(cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text, language="text")
        
        with tab2:
            if detected_keywords:
                st.write(f"**Mots-clés de phishing détectés ({len(detected_keywords)}):**")
                
                # Afficher les mots-clés sous forme de tags
                keyword_html = ""
                for keyword in detected_keywords:
                    keyword_html += f'<span class="keyword-tag">{keyword}</span> '
                
                st.markdown(f'<div style="margin: 10px 0;">{keyword_html}</div>', unsafe_allow_html=True)
                
                st.info(f"""
                **Interprétation:**
                - Ces mots sont souvent associés aux tentatives de phishing
                - Leur présence seule ne garantit pas un message malveillant
                - Le modèle IA considère le **contexte global**
                """)
            else:
                st.success("✅ Aucun mot-clé suspect détecté")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribution des probabilités:**")
                
                # Barres de progression
                st.progress(probabilities[0], text=f"Légitime: {probabilities[0]:.2%}")
                st.progress(probabilities[1], text=f"Phishing: {probabilities[1]:.2%}")
                
                st.write(f"**Décision finale:** {'PHISHING' if is_phishing else 'LÉGITIME'}")
                st.write(f"**Confiance:** {max(probabilities):.2%}")
            
            with col2:
                st.write("**Caractéristiques du texte:**")
                st.write(f"- Longueur originale: {len(original_text)} caractères")
                st.write(f"- Mots après nettoyage: {len(cleaned_text.split())}")
                st.write(f"- Mots-clés suspects: {len(detected_keywords)}")
                st.write(f"- Prédiction brute: {int(is_phishing)}")

# ============================================
# SECTION ÉDUCATIVE
# ============================================
def display_educational_section():
    """Affiche la section éducative"""
    st.markdown("---")
    st.subheader("🎓 Guide de détection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🚨 Signes typiques de phishing:</h4>
        <ul>
        <li><strong>Urgence artificielle:</strong> "URGENT", "IMMEDIAT", "ACTION REQUISE"</li>
        <li><strong>Demandes suspectes:</strong> informations personnelles, mots de passe</li>
        <li><strong>Liens dangereux:</strong> URLs raccourcies ou suspectes</li>
        <li><strong>Offres irréalistes:</strong> gains trop importants, prix gratuits</li>
        <li><strong>Expéditeur inconnu:</strong> adresse email ou numéro suspect</li>
        <li><strong>Erreurs nombreuses:</strong> fautes d'orthographe, grammaire incorrecte</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>✅ Bonnes pratiques:</h4>
        <ul>
        <li><strong>Vérifiez l'expéditeur:</strong> contacts fiables seulement</li>
        <li><strong>Survolez les liens:</strong> avant de cliquer</li>
        <li><strong>Méfiez-vous des urgences:</strong> vraies urgences sont rares</li>
        <li><strong>Contactez directement:</strong> l'organisation officielle</li>
        <li><strong>Utilisez 2FA:</strong> authentification à deux facteurs</li>
        <li><strong>Mettez à jour:</strong> vos logiciels régulièrement</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Avertissement important
    st.warning("""
    **⚠️ IMPORTANT:** Ce modèle a été entraîné sur des **emails en anglais**. 
    Pour les SMS ou textes en français, les résultats peuvent être **moins précis**. 
    Consultez toujours des sources officielles en cas de doute.
    """)

# ============================================
# FONCTION PRINCIPALE
# ============================================
def main():
    """Fonction principale de l'application"""
    # Initialisation
    setup_page()
    initialize_nltk()
    
    # Initialiser l'état de session
    if 'email_text' not in st.session_state:
        st.session_state.email_text = ""
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = False
    
    # Charger les modèles
    vectorizer, model, success = load_models()
    
    if not success:
        st.error("Impossible de charger les modèles. Vérifiez les fichiers .pkl")
        return
    
    # Afficher la sidebar
    display_sidebar()
    
    # Afficher le contenu principal
    display_main_content(vectorizer, model)

# ============================================
# POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    main()