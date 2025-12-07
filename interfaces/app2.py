# interfaces/app.py
import streamlit as st
import os
import sys

# Configuration de la page AVEC CSS POUR CACHER
st.set_page_config(
    page_title="Détecteur de Menaces - Accueil",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS pour cacher le menu Deploy et les 3 points
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Style pour les boutons */
    .stButton button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
        background: linear-gradient(135deg, #1f77b4 0%, #2c91d1 100%);
        color: white;
        border: none;
        padding: 12px 24px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Titre avec style
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4; margin-bottom: 10px;'>
        🛡️ Détecteur de Menaces
    </h1>
    <h3 style='text-align: center; color: #666; margin-bottom: 40px;'>
        Choisissez le type d'analyse :
    </h3>
    """, unsafe_allow_html=True)
    
    # Trois colonnes au lieu de deux
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔗 Analyse d'URLs")
        st.write("Détectez les sites malveillantes")
        if st.button("Tester une URL", key="url_btn", use_container_width=True):
            st.query_params = {"page": "url_analyzer"}
            st.rerun()
    
    with col2:
        st.markdown("### 📱 Analyse de SMS")
        st.write("Identifiez les SMS frauduleux ")
        if st.button("Tester un SMS", key="sms_btn", use_container_width=True):
            st.query_params = {"page": "sms_analyzer"}
            st.rerun()
    
    with col3:
        st.markdown("### 📧 Analyse de mail")
        st.write("Analyse complète: Contenu + URL")
        if st.button("Analyser un Email", key="mail_btn", use_container_width=True):
            st.query_params = {"page": "mail_analyzer"}
            st.rerun()
    
    # Vérifier si on doit afficher une autre page
    query_params = st.query_params
    
    if "page" in query_params:
        page = query_params["page"]
        
        if page == "url_analyzer":
            load_and_run_app("url2.py", "URL")
        
        elif page == "sms_analyzer":
            load_and_run_app("sms2.py", "SMS")
        
        elif page == "mail_analyzer":
            load_and_run_app("mail2.py", "Email Hybride")
    
    else:
        # Afficher la page d'accueil normale
        display_homepage()

def load_and_run_app(app_filename, app_name):
    """Charge et exécute une application spécifique"""
    try:
        # Ajouter le chemin actuel pour les imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Importer dynamiquement
        import importlib.util
        spec = importlib.util.spec_from_file_location(app_name.lower(), os.path.join(current_dir, app_filename))
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        # Exécuter la fonction main
        app_module.main()
        
        # Bouton de retour
        st.markdown("---")
        if st.button("⬅️ Retour à l'accueil", use_container_width=True, key=f"back_from_{app_name}"):
            st.query_params.clear()
            st.rerun()
            
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'application {app_name}: {e}")
        if st.button("⬅️ Retour", use_container_width=True, key=f"error_back_{app_name}"):
            st.query_params.clear()
            st.rerun()

def display_homepage():
    """Affiche le contenu de la page d'accueil"""
    # Informations supplémentaires
    st.markdown("---")
    st.markdown("### 📊 Performances du système")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Détection URLs", "96%", "Précision")
    with col2:
        st.metric("Détection SMS", "98%", "Précision")
    with col3:
        st.metric("Analyse Hybride", "99%", "Précision")
    
    # Section d'information
    st.markdown("---")
    st.markdown("### ℹ️ Comment utiliser")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        **🔗 Analyse d'URLs:**
        - Analyse technique des URLs
        - 48 caractéristiques examinées
        - Détection des domaines suspects
        - Idéal pour les liens seuls
        """)
    
    with col_info2:
        st.markdown("""
        **📱 Analyse de SMS:**
        - Analyse du contenu textuel
        - Détection des mots-clés suspects
        - Optimisé pour messages courts
        - Reconnaissance des patterns
        """)
    
    with col_info3:
        st.markdown("""
        **📧 Analyse Hybride:**
        - Combine contenu + URL
        - IA avancée (60%/40%)
        - Décision intelligente
        - Plus précis et fiable
        """)
    
    # Avantages de l'approche hybride
    st.markdown("---")
    st.markdown("### 🎯 Pourquoi choisir l'analyse hybride?")
    
    st.info("""
    **L'analyse hybride offre:**
    - ✅ **Plus de précision:** Combine les forces des deux modèles
    - ✅ **Moins de faux positifs:** Vérification croisée
    - ✅ **Analyse complète:** Contenu ET URL
    - ✅ **Décision expliquée:** Comprenez pourquoi
    
    **Recommandé pour:** Emails suspects, messages avec liens, communications importantes
    """)
    
    # Footer
    st.markdown("---")
    st.caption("🔒 Système de détection de menaces - Version 2.0 avec IA Hybride")

if __name__ == "__main__":
    main()