import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# --- Charger les données ---
data = pd.read_excel("new_als_train.xlsx")
data.columns = data.columns.str.strip()

# --- Colonnes ALSFRS-R ---
ALSFRS_questions = [
    "Q1 Speech","Q2 Salivation","Q3 Swallowing","Q4 Handwriting","Q5 Cutting",
    "Q6 Dressing and Hygiene","Q7 Turning in Bed","Q8 Walking","Q9 Climbing Stairs","Q10 Respiratory"
]

# --- Colonnes utilisées pour le modèle (ONSET retiré) ---
features = ["Age", "Weight", "Height", "Gender", 
            "Forced Vital Capacity", "Symptom Duration"] + ALSFRS_questions

# --- Préparer X et y ---
X = data[features].copy()
# Encodage seulement de Gender, pas Onset
X = pd.get_dummies(X, columns=["Gender"], drop_first=True)

y = data["Survived"].astype(int)  # 1 = survie ≥ 1 an, 0 = décès <1 an

# --- Séparer train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entraîner le modèle ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Évaluer ---
score = model.score(X_test, y_test)
print(f"Accuracy sur le test set : {score*100:.2f}%")

# --- Sauvegarder le modèle ---
joblib.dump(model, "model/rf_model.pkl")
print("Modèle sauvegardé dans 'model/rf_model.pkl'.")
