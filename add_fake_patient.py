import pandas as pd

# Charger les données existantes
data = pd.read_excel("new_als_train.xlsx")

# Définir un patient fictif
fake_patient = {
    "Age": 60,
    "Weight": 70,
    "Height": 170,
    "Gender": "Male",
    "Onset": "Limb",
    "Forced Vital Capacity": 65,
    "Symptom Duration": 12,
    "Q1 Speech": 3,
    "Q2 Salivation": 3,
    "Q3 Swallowing": 2,
    "Q4 Handwriting": 2,
    "Q5 Cutting": 2,
    "Q6 Dressing and Hygiene": 2,
    "Q7 Turning in Bed": 2,
    "Q8 Walking": 1,
    "Q9 Climbing Stairs": 1,
    "Q10 Respiratory": 2,
    "Survived": 'TRUE'
}

# Ajouter la ligne
data = pd.concat([data, pd.DataFrame([fake_patient])], ignore_index=True)

# Sauvegarder dans un nouveau fichier
data.to_excel("new_als_train_with_fake.xlsx", index=False)

print("Patient fictif ajouté avec succès ✅")
