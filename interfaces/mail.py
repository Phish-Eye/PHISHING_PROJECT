# interfaces/mail.py
import streamlit as st
import pickle
import numpy as np
import re
import string
import os
import sys
import nltk
from nltk.corpus import stopwords

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
def setup_page():
    """Configuration de la page"""
    st.set_page_config(
        page_title="Détecteur Hybride Email",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            padding-top: 1rem;
        }
        .hybrid-alert {
            background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            color: white;
            text-align: center;
            box-shadow: 0 4px 12px rgba(255, 94, 98, 0.3);
        }
        .safe-alert {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            color: white;
            text-align: center;
            box-shadow: 0 4px 12px rgba(86, 171, 47, 0.3);
        }
        .stButton button {
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Cacher le menu Deploy et les 3 points */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# FONCTION DE NETTOYAGE DE TEXTE
# ============================================
def fast_clean(text):
    """Nettoie le texte"""
    if not isinstance(text, str):
        return ""
    
    # Télécharger les stopwords si nécessaire
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ============================================
# FONCTION DE PRÉDICTION HYBRIDE
# ============================================
def hybrid_predict(content, url, hybrid_components):
    """Prédiction hybride directement dans l'application"""
    
    # Extraire les composants du modèle
    content_model = hybrid_components['content_model']
    content_vectorizer = hybrid_components['content_vectorizer']
    url_model = hybrid_components['url_model']
    url_scaler = hybrid_components['url_scaler']
    url_features = hybrid_components['url_features']
    
    # 1. Prédiction contenu
    cleaned_content = fast_clean(content)
    content_vectorized = content_vectorizer.transform([cleaned_content])
    prob_content = content_model.predict_proba(content_vectorized)[0][1]
    
    # 2. Prédiction URL
    # Extraire features basiques
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
    
    # Préparer les features dans le bon ordre
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
        'details': {
            'content_decision': 'PHISHING' if prob_content > 0.5 else 'LEGITIMATE',
            'url_decision': 'PHISHING' if prob_url > 0.5 else 'LEGITIMATE',
            'final_decision': 'PHISHING' if final_prob > 0.5 else 'LEGITIMATE'
        }
    }

# ============================================
# CHARGEMENT DU MODÈLE HYBRIDE (SILENCIEUX)
# ============================================
@st.cache_resource
def load_hybrid_model():
    """Charge le modèle hybride depuis le fichier pickle dans le même dossier"""
    try:
        # Chemin vers le fichier pickle (même dossier que mail_app.py)
        model_path = os.path.join(current_dir, 'hybrid_model_full.pkl')
        
        if not os.path.exists(model_path):
            return None
        
        # Charger le fichier pickle silencieusement
        with open(model_path, 'rb') as f:
            hybrid_components = pickle.load(f)
        
        # Vérifier que toutes les clés nécessaires sont présentes
        required_keys = ['content_model', 'content_vectorizer', 'url_model', 
                        'url_scaler', 'url_features']
        missing_keys = [key for key in required_keys if key not in hybrid_components]
        
        if missing_keys:
            return None
            
        return hybrid_components
        
    except Exception:
        return None

# ============================================
# SIDEBAR (CORRIGÉE - LES EXEMPLES FONCTIONNENT)
# ============================================
def display_sidebar():
    """Affiche la sidebar avec informations"""
    with st.sidebar:
        st.header("🤖 IA Hybride")
        st.markdown("""
        **Comment ça fonctionne :**
        - Combine l'analyse du **contenu** et de l'**URL**
        - Utilise 2 modèles IA différents
        - Prend une décision intelligente
        - Plus précis qu'un seul modèle
        """)
        
        st.markdown("---")
        st.header("🎯 Poids des modèles")
        
        st.markdown("""
        **Contenu textuel: 60%**
        - Analyse le message
        - Détecte l'intention
        - Reconnaît les patterns
        
        **URL: 40%**
        - Analyse la structure
        - Vérifie le domaine
        - Détecte les URLs suspectes
        """)
        
        st.markdown("---")
        st.header("📋 Tester rapidement")
        
        # Exemples - Nouvelle approche avec callback
        st.markdown("**Choisir un exemple :**")
        
        # Fonction callback pour mettre à jour les champs
        def set_example_content(example_num):
            """Met à jour le contenu selon l'exemple choisi"""
            if example_num == 1:
                # Exemple 1: Phishing bancaire
                st.session_state.mail_content = "URGENT: Your bank account has been suspended. Click here to verify your identity immediately."
                st.session_state.mail_url = "http://secure-bank-verification.com/login"
            elif example_num == 2:
                # Exemple 2: Email légitime Amazon
                st.session_state.mail_content = "Dear customer, your order #12345 has been shipped. You can track your package using the link below."
                st.session_state.mail_url = "https://www.amazon.com/track-order"
            elif example_num == 3:
                # Exemple 3: Cas limite
                st.session_state.mail_content = "Please update your payment information for your subscription renewal."
                st.session_state.mail_url = "http://billing-update.ga"
        
        # Exemple 1: Phishing bancaire
        if st.button("📧 Phishing bancaire", use_container_width=True, key="ex_phishing_btn"):
            set_example_content(1)
            # Rafraîchir la page pour montrer le contenu mis à jour
            st.rerun()
        
        # Exemple 2: Email légitime Amazon
        if st.button("📧 Email Amazon", use_container_width=True, key="ex_legit_btn"):
            set_example_content(2)
            st.rerun()
        
        # Exemple 3: Cas limite
        if st.button("📧 Mise à jour paiement", use_container_width=True, key="ex_limit_btn"):
            set_example_content(3)
            st.rerun()
        
        st.markdown("---")
        st.caption("🔒 Système de détection IA avancé")

# ============================================
# CONTENU PRINCIPAL
# ============================================
def display_main_content(hybrid_model):
    """Affiche le contenu principal de l'application"""
    # En-tête
    st.markdown('<h1 class="main-header">📧 Détecteur Hybride Email</h1>', unsafe_allow_html=True)
    st.markdown("### Analyse intelligente combinant contenu et URL")
    
    # Section d'analyse
    st.markdown("---")
    st.subheader("🔍 Analyse hybride")
    
    # Deux colonnes pour l'input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Contenu de l'email")
        
        # Initialiser si nécessaire
        if 'mail_content' not in st.session_state:
            st.session_state.mail_content = ""
        
        # Créer un text_area unique avec une clé stable
        content = st.text_area(
            "Email / Message:",
            value=st.session_state.mail_content,
            height=250,
            placeholder="Collez ici le contenu de l'email...\n\nExemple: URGENT: Your account needs verification. Click the link below to secure your account immediately.",
            key="content_input",
            help="Le contenu textuel à analyser"
        )
        # Mettre à jour le session_state si l'utilisateur tape
        if content != st.session_state.get('content_input_prev', ''):
            st.session_state.mail_content = content
            st.session_state.content_input_prev = content
    
    with col2:
        st.markdown("#### 🔗 URL de l'email")
        
        if 'mail_url' not in st.session_state:
            st.session_state.mail_url = ""
        
        # Créer un text_input unique avec une clé stable
        url = st.text_input(
            "URL à analyser:",
            value=st.session_state.mail_url,
            placeholder="https://secure-login-example.com",
            key="url_input",
            help="L'URL présente dans l'email"
        )
        # Mettre à jour le session_state si l'utilisateur tape
        if url != st.session_state.get('url_input_prev', ''):
            st.session_state.mail_url = url
            st.session_state.url_input_prev = url
        
        with st.expander("💡 Comment obtenir l'URL?"):
            st.write("""
            1. **Survolez** le lien avec la souris (sans cliquer)
            2. **Copiez** l'URL qui s'affiche
            3. **Collez-la** ici pour analyse
            4. **Ne cliquez jamais** sur un lien suspect!
            """)
    
    # Afficher ce qui est actuellement dans session_state (pour debug)
    if st.session_state.get('debug', False):
        st.write(f"Debug - Contenu: {st.session_state.mail_content[:50]}...")
        st.write(f"Debug - URL: {st.session_state.mail_url}")
    
    # Boutons d'action
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        analyze_btn = st.button(
            "🚀 Analyser avec IA Hybride",
            type="primary",
            use_container_width=True,
            disabled=(not st.session_state.mail_content or not st.session_state.mail_url),
            key="analyze_main_btn"
        )
    
    with col_btn2:
        clear_btn = st.button(
            "🧹 Effacer", 
            use_container_width=True,
            key="clear_btn"
        )
    
    with col_btn3:
        info_btn = st.button(
            "📊 Infos", 
            use_container_width=True,
            key="info_btn"
        )
    
    # Gestion des boutons
    if clear_btn:
        # Réinitialiser tous les champs
        st.session_state.mail_content = ""
        st.session_state.mail_url = ""
        st.session_state.content_input_prev = ""
        st.session_state.url_input_prev = ""
        st.rerun()
    
    if info_btn:
        st.session_state.show_info = not st.session_state.get('show_info', False)
        st.rerun()
    
    # Section d'information
    if st.session_state.get('show_info', False):
        with st.expander("📈 Informations techniques", expanded=True):
            if hybrid_model:
                st.write(f"**Modèle contenu:** {type(hybrid_model['content_model']).__name__}")
                st.write(f"**Modèle URL:** {type(hybrid_model['url_model']).__name__}")
                st.write(f"**Features URL analysées:** {len(hybrid_model['url_features'])}")
                st.write(f"**Poids:** 60% contenu, 40% URL")
                st.write("**Méthode:** Moyenne pondérée avec logique d'accord")
            else:
                st.write("Modèle non chargé")
    
    # ============================================
    # ANALYSE HYBRIDE
    # ============================================
    if analyze_btn and st.session_state.mail_content and st.session_state.mail_url:
        with st.spinner("🤖 Analyse hybride en cours..."):
            try:
                # Utiliser notre fonction de prédiction intégrée
                result = hybrid_predict(st.session_state.mail_content, st.session_state.mail_url, hybrid_model)
                
                # Affichage des résultats
                display_results(result)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")
                st.info("""
                **Dépannage:**
                1. Vérifiez que le modèle est correctement chargé
                2. Assurez-vous que NLTK est installé
                3. Redémarrez l'application
                """)
                import traceback
                st.code(traceback.format_exc())
    
    elif analyze_btn and (not st.session_state.mail_content or not st.session_state.mail_url):
        st.warning("⚠️ Veuillez remplir les deux champs (contenu et URL)")
    
    # ============================================
    # SECTION ÉDUCATIVE
    # ============================================
    st.markdown("---")
    display_educational_section()

# ============================================
# AFFICHAGE DES RÉSULTATS (SIMPLIFIÉ)
# ============================================
def display_results(result):
    """Affiche les résultats de l'analyse"""
    st.markdown("---")
    
    # Résultat final avec design adapté
    if result['is_phishing']:
        st.markdown(f"""
        <div class="hybrid-alert">
            <h2>🚨 PHISHING DÉTECTÉ PAR L'IA HYBRIDE</h2>
            <h3>Probabilité: {result['phishing_probability']:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommandations urgentes
        st.error("""
        **⚠️ ACTIONS IMMÉDIATES RECOMMANDÉES:**
        1. **NE CLIQUEZ PAS** sur les liens dans cet email
        2. **NE RÉPONDEZ PAS** à cet email
        3. **NE PARTAGEZ PAS** d'informations personnelles
        4. **SIGNALEZ** comme spam/phishing
        5. **CONTACTEZ** l'organisation via son site officiel
        """)
        
    else:
        st.markdown(f"""
        <div class="safe-alert">
            <h2>✅ EMAIL SÉCURISÉ</h2>
            <h3>Probabilité phishing: {result['phishing_probability']:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **✅ CET EMAIL SEMBLE SÉCURISÉ:**
        - Analyse du contenu: ✅ Passée
        - Analyse de l'URL: ✅ Passée
        - Décision IA hybride: ✅ Confiante
        """)
    
    # ============================================
    # ANALYSE DÉTAILLÉE SIMPLIFIÉE
    # ============================================
    st.markdown("---")
    st.subheader("📊 Analyse détaillée")
    
    # Métriques en colonnes
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    
    with col_met1:
        st.metric(
            "Probabilité finale",
            f"{result['phishing_probability']:.1%}",
            delta="PHISHING" if result['is_phishing'] else "SÉCURISÉ"
        )
    
    with col_met2:
        st.metric(
            "Analyse contenu",
            f"{result['content_prob']:.1%}",
            delta=result['details']['content_decision']
        )
    
    with col_met3:
        st.metric(
            "Analyse URL",
            f"{result['url_prob']:.1%}",
            delta=result['details']['url_decision']
        )
    
    with col_met4:
        agreement_icon = "✅" if result['agreement'] else "⚠️"
        st.metric(
            "Accord IA",
            agreement_icon,
            delta="D'ACCORD" if result['agreement'] else "DIVERGENT"
        )
    
    # ============================================
    # EXPLICATION DE LA DÉCISION (SIMPLIFIÉE)
    # ============================================
    with st.expander("🔍 Explication de la décision IA", expanded=True):
        
        # Barre de progression
        st.progress(result['phishing_probability'])
        st.caption(f"Niveau de risque: {result['phishing_probability']:.1%}")
        
        # Explication simple
        if result['agreement']:
            st.success(f"""
            **🤝 LES DEUX MODÈLES IA SONT D'ACCORD**
            - Les deux modèles indiquent: **{result['details']['final_decision']}**
            - Décision finale: **CONFIANTE**
            """)
        else:
            st.warning(f"""
            **⚖️ LES MODÈLES IA SONT EN DÉSACCORD**
            - Modèle contenu: **{result['details']['content_decision']}** ({result['content_prob']:.1%})
            - Modèle URL: **{result['details']['url_decision']}** ({result['url_prob']:.1%})
            - Décision basée sur la moyenne pondérée (60% contenu, 40% URL)
            """)
        
        st.write(f"**Seuil de décision:** 0.5 (50%)")
        st.write(f"**Résultat:** {'≥ 0.5 → PHISHING' if result['is_phishing'] else '< 0.5 → LÉGITIME'}")

# ============================================
# SECTION ÉDUCATIVE
# ============================================
def display_educational_section():
    """Affiche la section éducative"""
    st.subheader("🎓 Pourquoi l'IA hybride est plus efficace?")
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        st.markdown("""
        **✅ Avantages de l'approche hybride:**
        
        **1. Analyse complète**
        - Contenu: Ton, mots-clés, structure
        - URL: Caractéristiques techniques, domaine
        
        **2. Robustesse accrue**
        - Moins de faux positifs
        - Moins de faux négatifs
        - Décision plus fiable
        
        **3. Explications détaillées**
        - Compréhension des décisions
        - Transparence de l'IA
        - Aide à la prise de décision
        """)
    
    with col_edu2:
        st.markdown("""
        **🔧 Comment ça marche:**
        
        **Modèle 1: Analyse de contenu**
        - Détecte le phishing textuel
        - Optimisé pour emails anglais
        - Utilise TF-IDF + Logistic Regression
        
        **Modèle 2: Analyse d'URL**
        - Examine les caractéristiques techniques
        - Détecte les patterns de phishing
        - Utilise Random Forest
        
        **Combinaison intelligente**
        - Moyenne pondérée 60%/40%
        - Logique d'accord
        - Décision finale améliorée
        """)
    
    # FOOTER
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
        "🤖 Système de détection IA hybride • Contenu + URL"
        "</div>", 
        unsafe_allow_html=True
    )

# ============================================
# FONCTION PRINCIPALE
# ============================================
def main():
    """Fonction principale de l'application"""
    # Configuration
    setup_page()
    
    # Initialiser l'état de session AVANT d'afficher quoi que ce soit
    if 'mail_content' not in st.session_state:
        st.session_state.mail_content = ""
    if 'mail_url' not in st.session_state:
        st.session_state.mail_url = ""
    if 'content_input_prev' not in st.session_state:
        st.session_state.content_input_prev = ""
    if 'url_input_prev' not in st.session_state:
        st.session_state.url_input_prev = ""
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False
    
    # Charger le modèle SILENCIEUSEMENT
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    hybrid_model = load_hybrid_model()
    
    if not hybrid_model:
        st.error("""
        ❌ Modèle hybride non chargé.
        
        **Vérifiez que:**
        1. `hybrid_model_full.pkl` est dans le même dossier que `mail_app.py`
        2. Le fichier n'est pas corrompu
        """)
        st.stop()
    
    # Afficher la sidebar
    display_sidebar()
    
    # Afficher le contenu principal
    display_main_content(hybrid_model)

# ============================================
# POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    main()