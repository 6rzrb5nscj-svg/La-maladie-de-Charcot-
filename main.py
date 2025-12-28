import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =============================
# 1. CHARGER LES DONNÉES
# =============================
data = pd.read_excel("new_als_train.xlsx")

# Nettoyage des noms de colonnes
data.columns = data.columns.str.strip()

# =============================
# 2. VARIABLES
# =============================
ALSFRS_questions = [
    "Q1 Speech","Q2 Salivation","Q3 Swallowing","Q4 Handwriting","Q5 Cutting",
    "Q6 Dressing and Hygiene","Q7 Turning in Bed","Q8 Walking",
    "Q9 Climbing Stairs","Q10 Respiratory"
]

features = [
    "Age", "Weight", "Height", "Gender",
    "Forced Vital Capacity", "Symptom Duration"
] + ALSFRS_questions

# =============================
# 3. CRÉER X ET y
# =============================
X = data[features].copy()

# Encodage du genre uniquement (Onset supprimé)
X = pd.get_dummies(X, columns=["Gender"], drop_first=True)

y = data["Survived"].astype(int)

# =============================
# 4. TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# 5. MODÈLES
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(probability=True, random_state=42)
}

# =============================
# 6. ENTRAÎNEMENT & ÉVALUATION
# =============================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

# =============================
# 7. RÉSULTATS
# =============================
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print(results_df)

# =============================
# 8. SAUVEGARDE DU MEILLEUR MODÈLE
# =============================
best_model_name = results_df.iloc[0]["Model"]
joblib.dump(models[best_model_name], "model/best_model.pkl")

print(f"\nModèle sauvegardé : {best_model_name}")
