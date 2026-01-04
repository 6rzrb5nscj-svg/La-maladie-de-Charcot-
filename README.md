# Projet Prédiction de survie des patients SLA (ALS)

## Description
Ce projet permet de prédire la survie à 1 an des patients atteints de sclérose latérale amyotrophique (SLA / ALS) en utilisant des données cliniques, démographiques et fonctionnelles, notamment les scores ALSFRS-R.  
Le problème est formulé comme une classification binaire :

- 1 / True → survie ≥ 1 an  
- 0 / False → décès < 1 an  

Le projet permet :
1. D’entraîner un modèle de machine learning à partir des données cliniques
2. De sauvegarder et réutiliser le modèle entraîné
3. De prédire la survie d’un nouveau patient
4. D’utiliser une interface interactive pour tester les prédictions

---

## Structure du projet

```plaintext
├── model/
│   ├── rf_model.pkl
│   └── best_model.pkl
├── new_als_train.xlsx
├── main.py
├── app.py
├── train_model.py
├── add_fake_patient.py
└── README.md
```


## Données

Le dataset principal est un fichier Excel :
data/new_als_train.xlsx

Il contient des données cliniques et fonctionnelles de patients atteints de SLA.

### Colonnes utilisées

Données démographiques :
- Age
- Weight
- Height
- Gender

Données fonctionnelles :
- Forced Vital Capacity
- Symptom Duration

Scores ALSFRS-R :
- Q1 Speech
- Q2 Salivation
- Q3 Swallowing
- Q4 Handwriting
- Q5 Cutting
- Q6 Dressing and Hygiene
- Q7 Turning in Bed
- Q8 Walking
- Q9 Climbing Stairs
- Q10 Respiratory

Variable cible :
- Survived  
  - 1 → survie ≥ 1 an  
  - 0 → décès < 1 an  

Note importante :  
La variable Onset (Bulbaire / Spinal) a été volontairement exclue du modèle.  
Elle peut être affichée à titre informatif, mais n’est pas utilisée pour l’entraînement ni la prédiction.

---

## Modèle de machine learning

Le modèle utilisé est un Random Forest Classifier.

Paramètres principaux :
- Nombre d’arbres : 100
- Random state : 42
- Split : 80 % entraînement / 20 % test

Le Random Forest a été choisi pour sa robustesse, sa stabilité et sa capacité à gérer des variables hétérogènes sans normalisation stricte.

---

## Entraînement du modèle (main.py)

Le script main.py réalise les étapes suivantes :
1. Chargement des données depuis le fichier Excel
2. Nettoyage des noms de colonnes
3. Sélection des variables pertinentes
4. Encodage de la variable Gender
5. Séparation train / test
6. Entraînement du modèle Random Forest
7. Évaluation du modèle (accuracy)
8. Sauvegarde du modèle entraîné

Commande pour lancer l’entraînement :
python main.py

Le modèle entraîné est sauvegardé dans :
model/rf_model.pkl

---

## Application de prédiction (app.py)

Le fichier app.py permet :
- De charger le modèle sauvegardé
- De saisir les données d’un patient
- De prédire la survie à 1 an
- D’afficher le résultat de manière interactive

Lancement de l’application :
python app.py

ou avec Streamlit :
streamlit run app.py

---

## Patient fictif (add_fake_patient.py)

Le script add_fake_patient.py permet de :
- Créer un patient fictif
- Tester rapidement le modèle
- Vérifier le bon fonctionnement du pipeline de prédiction

Ce script est utile pour les tests, démonstrations et validations techniques.

---

## Installation

Prérequis :
- Python ≥ 3.9
- pip

Installation des dépendances :
pip install pandas scikit-learn openpyxl joblib streamlit

---

## Objectif du projet

L’objectif est de proposer un outil expérimental d’aide à la décision basé sur le machine learning pour estimer la probabilité de survie à 1 an chez des patients atteints de SLA à partir de données cliniques standardisées.

---

## Avertissement

Ce projet est strictement expérimental et académique.  
Les prédictions générées ne doivent en aucun cas être utilisées pour des décisions médicales réelles.  
Seul un professionnel de santé qualifié peut interpréter ces résultats dans un contexte clinique.

---

## Auteur

Projet réalisé dans un cadre data science / machine learning appliqué à la santé.
