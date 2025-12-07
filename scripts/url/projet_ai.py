import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import os

# Charger le dataset
path = os.path.join('..', '..', 'datasets', 'dataset_phishing.csv')
df = pd.read_csv(path)

# Sélection des features
features = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq',
    'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn',
    'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
    'ratio_digits_host', 'punycode', 'shortening_service', 'path_extension', 'phish_hints', 'domain_in_brand',
    'brand_in_subdomain', 'brand_in_path', 'suspecious_tld'
]

df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})

X = df[features]
y = df['status']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser les classificateurs (sans Gradient Boosting)
classifiers = {
    'AdaBoost': AdaBoostClassifier(),
    'LightGBM': LGBMClassifier()
}

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Cross-validation metrics
    cv_accuracy = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='precision')
    cv_recall = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='f1')

    results[name] = {
        'Accuracy': np.mean(cv_accuracy),
        'Precision': np.mean(cv_precision),
        'Recall': np.mean(cv_recall),
        'F1-Score': np.mean(cv_f1)
    }

# Affichage résultats
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [metrics['Accuracy'] for metrics in results.values()],
    'Precision': [metrics['Precision'] for metrics in results.values()],
    'Recall': [metrics['Recall'] for metrics in results.values()],
    'F1-Score': [metrics['F1-Score'] for metrics in results.values()]
})
print(metrics_df)

# Visualisation
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
