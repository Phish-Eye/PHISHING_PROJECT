import streamlit as st
import sys
import os
import pandas as pd

# Vérifier si la sidebar existe dans session state
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = True

def setup_page():
    """Configuration CSS pour forcer la sidebar visible"""
    st.markdown("""
    <style>
        /* Force la sidebar à être visible */
        section[data-testid="stSidebar"] {
            width: 300px !important;
            min-width: 300px !important;
            max-width: 300px !important;
            visibility: visible !important;
            transform: none !important;
        }
        
        /* Assure que le contenu principal a la bonne largeur */
        .main .block-container {
            max-width: calc(100vw - 300px);
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Styles existants */
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .phishing-alert {
            background-color: #ffcccc;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ff0000;
        }
        .safe-alert {
            background-color: #ccffcc;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00ff00;
        }
    </style>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Affiche la sidebar avec forçage si nécessaire"""
    # Créer explicitement la sidebar
    with st.sidebar:
        # Titre de la sidebar
        st.title("🔍 Analyse d'URL")
        
        # Bouton de retour
        if st.button("⬅️ Retour à l'accueil", use_container_width=True, type="secondary"):
            st.query_params.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Contenu de la sidebar
        st.header("ℹ️ Informations")
        st.markdown("""
        **Comment ça fonctionne :**
        - Analyse plus de 50 caractéristiques d'URL
        - Utilise l'IA pour détecter les patterns de phishing
        - Modèle Random Forest optimisé
        - Précision: >95%
        """)
        
        st.markdown("---")
        
        st.header("🧪 Exemples")
        st.markdown("**URLs suspectes :**")
        st.code("http://paypal-verification.tk\nhttps://facebook-login-security.ga", language=None)
        
        st.markdown("**URLs légitimes :**")
        st.code("https://www.google.com\nhttps://github.com", language=None)
        
        st.markdown("---")
        st.caption("Développé avec Python • Scikit-learn • Streamlit")

def display_main_content(predict_phishing_func):
    """Affiche le contenu principal"""
    # Titre principal
    st.markdown('<h1 class="main-header">🔍 Détecteur de Phishing Intelligent</h1>', unsafe_allow_html=True)
    
    # Interface principale
    col_title, col_back = st.columns([4, 1])
    with col_title:
        st.subheader("🔎 Analyse d'URL en temps réel")
    with col_back:
        if st.button("← Retour", key="back_btn_top"):
            st.query_params.clear()
            st.rerun()
    
    # Input URL
    url = st.text_input(
        "**Entrez l'URL à analyser :**",
        placeholder="https://example.com",
        key="url_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)
    
    # Analyse
    if analyze_btn and url:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                result = predict_phishing_func(url)
                if result:
                    display_results(result)
                else:
                    st.error("❌ Erreur lors de l'analyse")
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    elif analyze_btn and not url:
        st.warning("⚠️ Veuillez entrer une URL")
    
    # Section éducative (votre contenu existant)
    st.markdown("---")
    st.subheader("🎓 Comprendre la Détection de Phishing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Caractéristiques analysées :**")
        st.markdown("""
        - Longueur de l'URL
        - Nombre de points/tirets
        - Présence d'IP
        - Extension du domaine
        - Mots suspects
        """)
    
    with col2:
        st.info("**🚨 Signaux d'alerte :**")
        st.markdown("""
        - URLs très longues
        - Nombreux caractères spéciaux
        - Domaines suspects (.tk, .ml)
        - Mots 'login', 'verify', 'secure'
        - Services de raccourcissement
        """)
    
    with col3:
        st.info("**✅ Bonnes pratiques :**")
        st.markdown("""
        - Vérifiez toujours l'URL
        - Méfiez-vous des emails suspects
        - Utilisez l'authentification 2FA
        - Vérifiez le certificat SSL
        """)

def display_results(result):
    """Affiche les résultats (gardez votre code existant)"""
    # ... votre code existant pour display_results ...

def load_models():
    """Charge les modèles (gardez votre code existant)"""
    # ... votre code existant pour load_models ...

def main():
    """Fonction principale"""
    setup_page()
    
    # Charger les modèles
    success, load_model_func, predict_func = load_models()
    if not success:
        st.stop()
    
    # Initialisation du modèle
    @st.cache_resource
    def load_model():
        if load_model_func():
            return True
        return False
    
    if 'model_loaded' not in st.session_state:
        if load_model():
            st.session_state.model_loaded = True
        else:
            st.error("❌ Erreur de chargement du modèle")
            st.stop()
    
    # Afficher la sidebar (forcée)
    display_sidebar()
    
    # Afficher le contenu principal
    display_main_content(predict_func)
    
    # Footer
    st.markdown("---")
    st.caption("🔒 Système de détection de phishing v2.0")

if __name__ == "__main__":
    main()