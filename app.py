import streamlit as st
import pandas as pd
import joblib

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prédiction survie SLA",
    page_icon="🧬",
    layout="centered"
)

# --- Charger le modèle Gradient Boosting entraîné ---
model = joblib.load("model/best_model.pkl")

# --- Questions ALSFRS-R ---
ALSFRS_questions = [
    "Q1 Speech", "Q2 Salivation", "Q3 Swallowing", "Q4 Handwriting", "Q5 Cutting",
    "Q6 Dressing and Hygiene", "Q7 Turning in Bed", "Q8 Walking",
    "Q9 Climbing Stairs", "Q10 Respiratory"
]

ALSFRS_labels_fr = {
    "Q1 Speech": "Parole",
    "Q2 Salivation": "Salivation",
    "Q3 Swallowing": "Déglutition",
    "Q4 Handwriting": "Écriture",
    "Q5 Cutting": "Découpage des aliments",
    "Q6 Dressing and Hygiene": "Habillage et hygiène",
    "Q7 Turning in Bed": "Se retourner dans le lit",
    "Q8 Walking": "Marche",
    "Q9 Climbing Stairs": "Monter les escaliers",
    "Q10 Respiratory": "Fonction respiratoire"
}

# --- Charger dataset pour récupérer les colonnes du modèle ---
data = pd.read_excel("new_als_train.xlsx")
data.columns = data.columns.str.strip()
features = ["Age", "Weight", "Height", "Gender", 
            "Forced Vital Capacity", "Symptom Duration"] + ALSFRS_questions
X = data[features].copy()
# Encodage uniquement Gender pour correspondre au modèle
X_encoded = pd.get_dummies(X, columns=["Gender"], drop_first=True)

# --- Titre ---
st.title("Prédiction de survie à 1 an pour un patient SLA")

# --- Note explicative ---
st.markdown("""
    <style>
    .note-alsfrs {
        position: fixed;
        top: 80px;
        right: 20px;
        width: 230px;
        background-color: #8AA5CF;
        border-left: 5px solid #1D3E5C;
        padding: 12px 15px;
        border-radius: 8px;
        font-size: 14px;
        color: #000000;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        z-index: 9999;
    }
    </style>

    <div class="note-alsfrs">
        <strong>Note aux utilisateurs pour le score ALSFRS :</strong><br>
        De 0 (fonction sévèrement altérée) à 4 (fonction normale / préservée)
    </div>
""", unsafe_allow_html=True)

# --- Formulaire patient ---
with st.form("patient_form"):
    st.subheader("Informations patient")
    Age = st.number_input("Âge", 18, 120, 60)
    Weight = st.number_input("Poids (kg)", 30, 200, 70)
    Height = st.number_input("Taille (cm)", 100, 250, 170)
    Gender = st.selectbox("Sexe", ["Male", "Female"])

    st.subheader("ALSFRS-R - Sous-scores")
    patient_alsfrs = {}
    for q in ALSFRS_questions:
        patient_alsfrs[q] = st.slider(
            ALSFRS_labels_fr[q],
            0, 4, 4
        )

    Forced_VC = st.number_input(
        "Capacité vitale forcée (échelle 0-7, 7 = normale)",
        0, 7, 7
    )

    Symptom_Duration = st.number_input(
        "Durée des symptômes (mois)",
        0, 120, 0
    )

    submitted = st.form_submit_button("Prédire la survie")

# --- Prédiction ---
if submitted:
    # Créer DataFrame patient (Onset exclu)
    patient_data = pd.DataFrame({
        "Age": [Age],
        "Weight": [Weight],
        "Height": [Height],
        "Gender": [Gender],
        "Forced Vital Capacity": [Forced_VC],
        "Symptom Duration": [Symptom_Duration],
        **patient_alsfrs
    })

    # Encodage Gender seulement
    patient_encoded = pd.get_dummies(patient_data, columns=["Gender"], drop_first=True)

    # Ajouter colonnes manquantes pour correspondre au modèle
    for col in X_encoded.columns:
        if col not in patient_encoded.columns:
            patient_encoded[col] = 0

    # Réordonner les colonnes
    patient_encoded = patient_encoded[X_encoded.columns]

    # Prédiction
    pred = model.predict(patient_encoded)[0]
    prob = model.predict_proba(patient_encoded)[0][1]

    # Déterminer couleur et label
    if prob < 0.33:
        color = "#d9534f"
        risk_label = "Faible chance de survie à 1 an"
    elif prob < 0.66:
        color = "#f0ad4e"
        risk_label = "Chance modérée de survie à 1 an"
    else:
        color = "#5cb85c"
        risk_label = "Forte chance de survie à 1 an"

    # Affichage
    st.markdown(f"""
        <div style="background-color:{color};padding:15px;border-radius:8px;color:white;text-align:center;">
            <h3>{risk_label}</h3>
            <p>Probabilité de survie à 1 an : <strong>{prob*100:.1f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

# --- Disclaimer ---
st.markdown("---")
st.info(
    "⚠️ Ce modèle est uniquement un outil d'aide à la décision. "
    "Il ne remplace pas l'avis d'un professionnel de santé. "
    "Les résultats doivent être interprétés avec prudence et dans le contexte médical approprié."
)

