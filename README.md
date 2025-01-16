# Simplon_Machine_learning
# Prédire une Prime d'Assurance : Assur'Aimant US Expansion

Ce projet vise à aider Assur'Aimant, un assureur français, à estimer les primes d'assurance pour son expansion aux États-Unis.  Actuellement, l'estimation manuelle des primes est coûteuse et prend du temps.  Ce projet utilise le Machine Learning pour prédire les primes en fonction des données démographiques des clients.

## Contexte du Projet

Assur'Aimant souhaite moderniser son processus d'estimation des primes d'assurance pour le marché américain.  Nous avons été mandatés pour développer une solution d'IA capable de prédire avec précision les primes en fonction des caractéristiques des clients.  Ce projet comprend une analyse exploratoire des données (EDA) et la construction d'un modèle prédictif.

## Données

Les données collectées auprès d'Assur'Aimant à Houston comprennent les informations suivantes :

- **`BMI`**: Indice de Masse Corporelle (18.5 - 24.9 idéalement).
- **`Sex`**: Sexe du souscripteur (homme ou femme).
- **`Age`**: Âge du bénéficiaire principal.
- **`Children`**: Nombre d'enfants à charge couverts par l'assurance.
- **`Smoker`**:  Statut fumeur (fumeur ou non-fumeur).
- **`Region`**: Région de résidence aux États-Unis (Nord-Est, Sud-Est, Sud-Ouest, Nord-Ouest).
- **`Charges`**: Prime d'assurance facturée (variable cible).


## Objectifs

1. **Analyse Exploratoire des Données (EDA):**  Comprendre les données, identifier les tendances, les valeurs aberrantes et les relations entre les variables.  Ceci comprend :

    - Vérification des valeurs manquantes et des doublons (avec `missingno`).
    - Détection des valeurs aberrantes.
    - Analyse univariée et bivariée.
    - Analyse de corrélation.
    - Validation d'hypothèses avec des tests statistiques.
    - Visualisations avec `seaborn` (box plots, violin plots, etc.).

2. **Modélisation Prédictive:**  Construire un modèle de Machine Learning pour prédire les primes d'assurance.  Ceci comprend :

    - Création d'un modèle de base (Dummy Model).
    - Séparation des données (80% entraînement, 20% test).
    - Préparation des données (transformation logarithmique si nécessaire, gestion des `random_state` et `seed`).
    - Sélection de modèles (`sklearn`: Régression Linéaire, Lasso, Ridge, ElasticNet).
    - Évaluation des modèles (R², RMSE).
    - Pré-traitement (Standardisation, encodage des variables catégorielles avec `sklearn.pipeline.Pipeline`).
    - Optimisation ( `PolynomialFeatures`, `GridSearchCV`, `RandomSearchCV`).
    - Analyse et interprétation des résultats (importance des variables).

3. **Application Streamlit:** Développer une application interactive permettant :

    - La saisie des données par l'utilisateur.
    - La prédiction des primes en temps réel.
    - L'utilisation d'un modèle pré-entraîné exporté en `.pkl`.
    - L'intégration des pipelines de pré-traitement.


## Outils et Technologies

- Python
- `pandas`, `numpy`
- `scikit-learn`
- `seaborn`, `missingno`
- `streamlit`


## Structure du Projet



## Installation


```bash
git clone https://github.com/votre_utilisateur/votre_repo.git

cd votre_repo

pip install -r [requirements.txt](VALID_FILE)
```

Exécution de l'application Streamlit

```bash
streamlit run [app.py](VALID_FILE)
```

```mermaid
graph TD
    A[Collecte des données] --> B(Analyse exploratoire);
    B --> C{Choix du modèle};
    C -- Régression linéaire --> D[Evaluation];
    C -- Lasso --> D;
    C -- Ridge --> D;
    C -- ElasticNet --> D;
    D --> E[Optimisation];
    E --> F[Application Streamlit];

