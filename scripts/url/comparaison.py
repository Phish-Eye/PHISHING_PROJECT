import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle

# ==================== PARTIE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES ====================

# Chargement du dataset
path = os.path.join('..', '..', 'datasets', 'dataset_phishing.csv')
df = pd.read_csv(path)

print("Dataset chargé avec succès!")
print(f"Dimensions: {df.shape}")
print(f"Nombre de features: {len(df.columns)}")

# Conversion de la variable cible
mapping = {'legitimate': 0, 'phishing': 1}
df['status'] = df['status'].map(mapping)

print(f"\nDistribution des classes:")
print(df['status'].value_counts())
print(f"Ratio phishing: {df['status'].mean():.3f}")

# ==================== PARTIE 2: ANALYSE DE CORRÉLATION ET SÉLECTION DES FEATURES ====================

# Liste COMPLÈTE des features qui peuvent être extraites d'une URL sans scraping profond
EXTRACTABLE_FEATURES = [
    # Features basiques de l'URL
    'length_url', 'length_hostname', 'ip',
    
    # Compteurs de caractères dans l'URL
    'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_eq', 
    'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star',
    'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space',
    'nb_www', 'nb_com', 'nb_dslash',
    
    # Features structurelles
    'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host',
    'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain',
    'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service',
    'path_extension', 'nb_redirection', 'nb_external_redirection',
    
    # Features de mots (peuvent être estimées)
    'length_words_raw', 'char_repeat', 'shortest_words_raw', 'shortest_word_host',
    'shortest_word_path', 'longest_words_raw', 'longest_word_host', 'longest_word_path',
    'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints',
    
    # Features de marque (peuvent être estimées)
    'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld',
    
    # Features techniques (valeurs par défaut réalistes)
    'statistical_report', 'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks',
    'ratio_nullHyperlinks', 'nb_extCSS', 'ratio_intRedirection', 'ratio_extRedirection', 
    'ratio_intErrors', 'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags',
    'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'popup_window',
    'safe_anchor', 'onmouseover', 'right_clic', 'empty_title', 'domain_in_title',
    'domain_with_copyright',
    
    # Features WHOIS et trafic (estimations intelligentes)
    'whois_registered_domain', 'domain_registration_length', 'domain_age',
    'web_traffic', 'dns_record', 'google_index', 'page_rank'
]

# Filtrer pour garder seulement les features disponibles dans le dataset
available_features = [f for f in EXTRACTABLE_FEATURES if f in df.columns]
print(f"\nFeatures extractibles disponibles: {len(available_features)}")

# Analyser la corrélation avec la target
corr_matrix = df[available_features + ['status']].corr(numeric_only=True)
target_corr = corr_matrix['status'].abs().sort_values(ascending=False)

print(f"\nTop 15 features les plus corrélées:")
for i, (feature, corr) in enumerate(target_corr.head(15).items(), 1):
    if feature != 'status':
        print(f"  {i:2d}. {feature}: {corr:.3f}")

# Sélectionner les features les plus importantes (corrélation > 0.1)
important_features = target_corr[target_corr > 0.1].index.tolist()
if 'status' in important_features:
    important_features.remove('status')

print(f"\nFeatures importantes sélectionnées (corrélation > 0.1): {len(important_features)}")
for i, feature in enumerate(important_features, 1):
    corr_value = target_corr[feature]
    print(f"  {i:2d}. {feature}: {corr_value:.3f}")

# ==================== PARTIE 3: MATRICE DE CORRÉLATION DES FEATURES IMPORTANTES ====================

print("\n=== MATRICE DE CORRÉLATION DES FEATURES IMPORTANTES ===")

# Prendre les 15 features les plus corrélées pour la visualisation
top_corr_features = target_corr.head(16).index.tolist()  # Inclut 'status'
if 'status' in top_corr_features:
    top_corr_features.remove('status')
top_corr_features = top_corr_features[:15]  # Garder les 15 meilleures

top_corr_matrix = df[top_corr_features + ['status']].corr(numeric_only=True)

plt.figure(figsize=(16, 14))
sns.heatmap(top_corr_matrix, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8},
            annot_kws={'size': 10, 'weight': 'bold'})

plt.title('Matrice de Corrélation - Top 15 Features les plus Corrélées (|corr| > 0.1)\n', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Préparation des données pour l'entraînement
X = df[important_features]
y = df['status']

# ==================== PARTIE 4: OPTIMISATION DES MODÈLES ====================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n=== RÉPARTITION DES DONNÉES ===")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimisation des hyperparamètres
print("\n=== OPTIMISATION DES MODÈLES ===")

# Random Forest optimisé
print("Optimisation de Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

print(f"RF meilleurs paramètres: {rf_grid.best_params_}")

# SVM optimisé
print("Optimisation de SVM...")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

svm = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm, svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_

print(f"SVM meilleurs paramètres: {svm_grid.best_params_}")

# Regression Logistique optimisée
print("Optimisation de Regression Logistique...")
lr_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

lr = LogisticRegression(random_state=42, max_iter=1000)
lr_grid = GridSearchCV(lr, lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_

print(f"LR meilleurs paramètres: {lr_grid.best_params_}")

# Naive Bayes (pas d'optimisation nécessaire)
nb_best = GaussianNB()
nb_best.fit(X_train_scaled, y_train)

# ==================== PARTIE 5: ÉVALUATION DES MODÈLES ====================

models = {
    'Random Forest': rf_best,
    'SVM': svm_best,
    'Regression Logistique': lr_best,
    'Naive Bayes': nb_best
}

results = {}

print("\n=== ÉVALUATION DES MODÈLES OPTIMISÉS ===")

for name, model in models.items():
    if name in ['SVM', 'Regression Logistique', 'Naive Bayes']:
        X_train_eval = X_train_scaled
        X_test_eval = X_test_scaled
    else:
        X_train_eval = X_train
        X_test_eval = X_test
    
    # Prédictions
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")

# ==================== PARTIE 6: MATRICES DE CONFUSION SÉPARÉES ====================

print("\n=== MATRICES DE CONFUSION SÉPARÉES ===")

for name, result in results.items():
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, result['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    
    plt.title(f'{name} - Matrice de Confusion\nAccuracy: {result["accuracy"]:.3f} | F1-Score: {result["f1"]:.3f}', 
              fontweight='bold', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Comparaison finale
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'AUC': [results[m]['auc'] for m in results.keys()]
}).sort_values('Accuracy', ascending=False)

print("\n=== COMPARAISON FINALE ===")
print(comparison_df)

# Graphique de comparaison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'F1-Score', 'AUC']
x = np.arange(len(comparison_df))
width = 0.25

for i, metric in enumerate(metrics):
    values = comparison_df[metric].values
    plt.bar(x + i*width, values, width, label=metric, alpha=0.8)

plt.xlabel('Modèles')
plt.ylabel('Scores')
plt.title('Comparaison des Performances des Modèles Optimisés', fontweight='bold')
plt.xticks(x + width, comparison_df['Model'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)

for i, (acc, f1, auc) in enumerate(zip(comparison_df['Accuracy'], comparison_df['F1-Score'], comparison_df['AUC'])):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.text(i + width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.text(i + 2*width, auc + 0.02, f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

