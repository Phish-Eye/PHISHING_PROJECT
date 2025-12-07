import numpy as np
import pandas as pd
import urllib.parse
import tldextract
import re

def extract_url_features(url):
    """
    Extrait les features d'une nouvelle URL en utilisant les mêmes calculs que le dataset
    """
    try:
        # Nettoyer l'URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urllib.parse.urlparse(url)
        domain_info = tldextract.extract(url)
        hostname = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        features = {}
        
        # 1. Features basiques
        features['length_url'] = len(url)
        features['length_hostname'] = len(hostname)
        
        # 2. Détection IP
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features['ip'] = 1 if re.search(ip_pattern, hostname) else 0
        
        # 3. Compteurs de caractères
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_tilde'] = url.count('~')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')
        features['nb_www'] = 1 if hostname.startswith('www.') else 0
        features['nb_com'] = 1 if hostname.endswith('.com') else 0
        features['nb_dslash'] = url.count('//')
        
        # 4. Features structurelles
        features['http_in_path'] = 1 if 'http' in path else 0
        features['https_token'] = 1 if 'https' in url else 0
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / max(len(url), 1)
        features['ratio_digits_host'] = sum(c.isdigit() for c in hostname) / max(len(hostname), 1)
        features['punycode'] = 1 if 'xn--' in hostname else 0
        features['port'] = 1 if ':' in hostname and hostname.rfind(':') > hostname.rfind(']') else 0
        features['tld_in_path'] = 1 if domain_info.suffix in path else 0
        features['tld_in_subdomain'] = 1 if domain_info.suffix in domain_info.subdomain else 0
        features['abnormal_subdomain'] = 1 if len(domain_info.subdomain) > 3 else 0
        features['nb_subdomains'] = hostname.count('.')
        features['prefix_suffix'] = 1 if re.search(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
        features['random_domain'] = 1 if len(domain_info.domain) < 4 else 0
        
        # 5. Détection des services de raccourcissement
        shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 
                              'buff.ly', 'adf.ly', 'bitly.com', 'shorturl.at', 'cutt.ly']
        features['shortening_service'] = 1 if any(service in hostname for service in shortening_services) else 0
        
        features['path_extension'] = 1 if '.' in path.split('/')[-1] else 0
        
        # 6. Estimations pour les features complexes
        features['nb_redirection'] = url.count('redirect') + url.count('url=')
        features['nb_external_redirection'] = features['nb_redirection']  # Estimation
        
        # 7. Features de mots (estimations)
        words = re.findall(r'[a-zA-Z]+', url)
        features['length_words_raw'] = len(''.join(words))
        
        # char_repeat - plus longue séquence de caractères répétés
        char_repeat = 0
        for char in set(url):
            char_count = 0
            max_char_count = 0
            for c in url:
                if c == char:
                    char_count += 1
                    max_char_count = max(max_char_count, char_count)
                else:
                    char_count = 0
            char_repeat = max(char_repeat, max_char_count)
        features['char_repeat'] = char_repeat
        
        features['shortest_words_raw'] = min([len(w) for w in words]) if words else 0
        features['shortest_word_host'] = min([len(w) for w in re.findall(r'[a-zA-Z]+', hostname)]) if hostname else 0
        features['shortest_word_path'] = min([len(w) for w in re.findall(r'[a-zA-Z]+', path)]) if path else 0
        features['longest_words_raw'] = max([len(w) for w in words]) if words else 0
        features['longest_word_host'] = max([len(w) for w in re.findall(r'[a-zA-Z]+', hostname)]) if hostname else 0
        features['longest_word_path'] = max([len(w) for w in re.findall(r'[a-zA-Z]+', path)]) if path else 0
        features['avg_words_raw'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_word_host'] = np.mean([len(w) for w in re.findall(r'[a-zA-Z]+', hostname)]) if hostname else 0
        features['avg_word_path'] = np.mean([len(w) for w in re.findall(r'[a-zA-Z]+', path)]) if path else 0
        
        # 8. Indices de phishing
        phishing_terms = ['login', 'secure', 'account', 'verify', 'banking', 'update', 'signin']
        features['phish_hints'] = sum(1 for term in phishing_terms if term in url.lower())
        
        # 9. Features de marque (estimations)
        features['domain_in_brand'] = 1 if len(domain_info.domain) > 5 else 0
        features['brand_in_subdomain'] = 1 if any(brand in domain_info.subdomain for brand in ['paypal', 'google', 'facebook']) else 0
        features['brand_in_path'] = 1 if any(brand in path for brand in ['paypal', 'google', 'facebook']) else 0
        features['suspecious_tld'] = 1 if domain_info.suffix in ['.tk', '.ml', '.ga', '.cf'] else 0
        
        # 10. Features techniques (valeurs par défaut réalistes)
        features['statistical_report'] = 0
        features['nb_hyperlinks'] = 10  # Valeur moyenne
        features['ratio_intHyperlinks'] = 0.8
        features['ratio_extHyperlinks'] = 0.2
        features['ratio_nullHyperlinks'] = 0.0
        features['nb_extCSS'] = 1
        features['ratio_intRedirection'] = 0.0
        features['ratio_extRedirection'] = 0.0
        features['ratio_intErrors'] = 0.0
        features['ratio_extErrors'] = 0.0
        features['login_form'] = 1 if any(term in url for term in ['login', 'signin']) else 0
        features['external_favicon'] = 0
        features['links_in_tags'] = 5  # Valeur moyenne
        features['submit_email'] = 0
        features['ratio_intMedia'] = 0.8
        features['ratio_extMedia'] = 0.2
        features['sfh'] = 0
        features['iframe'] = 0
        features['popup_window'] = 0
        features['safe_anchor'] = 1
        features['onmouseover'] = 0
        features['right_clic'] = 0
        features['empty_title'] = 0
        features['domain_in_title'] = 1
        features['domain_with_copyright'] = 0
        
        # 11. Features WHOIS et trafic (estimations intelligentes)
        well_known_domains = ['google', 'facebook', 'youtube', 'amazon', 'microsoft', 
                             'apple', 'wikipedia', 'twitter', 'instagram']
        if any(domain in hostname for domain in well_known_domains):
            features['whois_registered_domain'] = 1
            features['domain_registration_length'] = 3650  # 10 ans
            features['domain_age'] = 5000  # Ancien
            features['web_traffic'] = 1000000  # Élevé
            features['dns_record'] = 1
            features['google_index'] = 1
            features['page_rank'] = 8
        else:
            features['whois_registered_domain'] = 1
            features['domain_registration_length'] = 365  # 1 an
            features['domain_age'] = 365  # 1 an
            features['web_traffic'] = 1000  # Faible
            features['dns_record'] = 1
            features['google_index'] = 0
            features['page_rank'] = 2
        
        print(f"✅ {len(features)} features extraites pour: {url}")
        return features
        
    except Exception as e:
        print(f"❌ Erreur extraction features: {e}")
        return None