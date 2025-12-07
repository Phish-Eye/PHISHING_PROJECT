import streamlit as st
import sys
import os
import pandas as pd
if "_sidebar_state" in st.session_state:
    del st.session_state["_sidebar_state"]

def setup_page():
    """Configuration de la page"""
    st.set_page_config(
        page_title="Détecteur de Phishing",
        page_icon="🔍",
        layout="wide",
        
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
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
        .feature-card {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)

def load_models():
    """Charge les modèles nécessaires"""
    # Ajouter le chemin des scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_path = os.path.join(current_dir, '..', 'scripts', 'url')
    sys.path.append(scripts_path)

    try:
        from load_model import load_phishing_model, predict_phishing
        return True, load_phishing_model, predict_phishing
    except ImportError as e:
        st.error(f"❌ Erreur d'importation: {e}")
        return False, None, None

def display_sidebar():
    """Affiche la sidebar"""
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.markdown("""
        **Comment ça fonctionne :**
        - Analyse plus de 50 caractéristiques d'URL
        - Utilise l'IA pour détecter les patterns de phishing
        - Modèle Random Forest optimisé
        - Précision: >95%
        
        **Développé avec :**
        - Python • Scikit-learn • Streamlit
        """)
        
        st.header("🧪 Exemples de test")
        st.markdown("""
        **URLs suspectes :**
        ```
        http://paypal-verification.tk
        https://facebook-login-security.ga
        http://amazon-update-account.ml
        ```
        
        **URLs légitimes :**
        ```
        https://www.google.com
        https://github.com
        https://stackoverflow.com
        ```
        """)

def display_main_content(predict_phishing_func):
    """Affiche le contenu principal"""
    # Titre principal
    st.markdown('<h1 class="main-header">🔍 Détecteur de Phishing Intelligent</h1>', unsafe_allow_html=True)
    
    # Interface principale
    st.subheader("🔎 Analyse d'URL en temps réel")

    # Input URL
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input(
            "Entrez l'URL à analyser :",
            placeholder="https://example.com",
            key="url_input"
        )

    with col2:
        st.write("")  # Espacement
        st.write("")
        analyze_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)

    # Section d'analyse
    if analyze_btn and url:
        with st.spinner("🔍 Analyse en cours... Cette opération peut prendre quelques secondes"):
            result = predict_phishing_func(url)
        
        if result:
            display_results(result)
        else:
            st.error("❌ Erreur lors de l'analyse de l'URL. Vérifiez le format de l'URL.")

    elif analyze_btn and not url:
        st.warning("⚠️ Veuillez entrer une URL à analyser")

    # Section éducative
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
    """Affiche les résultats de l'analyse"""
    probability = result['probability'] * 100
    confidence = result['confidence']
    
    # Affichage du résultat principal
    st.markdown("---")
    
    if result['prediction'] == 1:
        st.markdown(f"""
        <div class="phishing-alert">
            <h2>🚨 PHISHING DÉTECTÉ</h2>
            <h3>Risque de phishing : {probability:.1f}%</h3>
            <p><strong>Confiance :</strong> {confidence} | <strong>Modèle :</strong> {result['model_used']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ Cette URL présente des caractéristiques suspectes. Ne saisissez pas d'informations personnelles !")
        
    else:
        st.markdown(f"""
        <div class="safe-alert">
            <h2>✅ URL SÉCURISÉE</h2>
            <h3>Risque de phishing : {probability:.1f}%</h3>
            <p> <strong>Modèle :</strong> {result['model_used']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("🎉 Cette URL semble légitime. Vous pouvez naviguer en toute confiance.")
    
    # Détails techniques
    with st.expander("📊 DÉTAILS TECHNIQUES DES FEATURES", expanded=True):
        st.write(f"**URL analysée :** `{result['url']}`")
        
        # Features sous forme de métriques
        st.subheader("📈 Métriques importantes")
        
        # Sélectionner les features les plus significatives
        important_features = {
            k: v for k, v in result['features'].items() 
            if v != 0 or k in ['length_url', 'nb_dots', 'nb_hyphens']
        }
        
        # Afficher en colonnes
        cols = st.columns(3)
        for i, (feature, value) in enumerate(important_features.items()):
            with cols[i % 3]:
                if value > 0 and feature not in ['length_url', 'nb_dots']:
                    st.metric(
                        label=feature.replace('_', ' ').title(),
                        value=value,
                        delta="⚠️ Risque" if value > 0 else None
                    )
                else:
                    st.metric(
                        label=feature.replace('_', ' ').title(),
                        value=value
                    )
    
    # Graphique de probabilité
    with st.expander("📊 VISUALISATION DU RISQUE"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Jauge de risque
            risk_level = "Élevé" if probability > 70 else "Moyen" if probability > 30 else "Faible"
            
            st.metric(
                label="Niveau de Risque",
                value=risk_level,
                delta=f"{probability:.1f}%"
            )
        
        with col2:
            # Barre de progression
            st.progress(probability / 100)
            st.caption(f"Probabilité de phishing: {probability:.1f}%")

def main():
    """Fonction principale"""
    setup_page()
    
    # Charger les modèles
    success, load_model_func, predict_func = load_models()
    if not success:
        st.stop()
    
    # Charger le modèle
    @st.cache_resource
    def load_phishing_model_cached():
        if load_model_func():
            return True
        return False

    # Initialisation
    if 'model_loaded' not in st.session_state:
        #st.info("🔄 Chargement du modèle de phishing...")
        if load_phishing_model_cached():
            st.session_state.model_loaded = True
            #st.success("✅ Modèle chargé avec succès!")
        else:
            st.error("❌ Erreur lors du chargement du modèle")
            st.error("💡 Exécutez d'abord : `python train_model.py` dans le dossier scripts/url/")
            st.stop()
    
    # Afficher la sidebar
    display_sidebar()
    
    # Afficher le contenu principal
    display_main_content(predict_func)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔒 Système de détection de phishing - Développé avec Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()