import os
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import shap

app = Flask(__name__)

THRESHOLD = 0.6194837602241042

# Répertoire du fichier api.py :
current_directory = os.path.dirname(os.path.abspath(__file__))

# Chargement du modèle :
model_path = os.path.join(current_directory, "model.pkl")
model = joblib.load(model_path)
feature_names = model.named_steps['classifier'].feature_names_

@app.route('/predict', methods=['POST'])
def predict():
    # Récupération de l'identifiant du client à partir de la requête :
    data_dict = request.json

    # Remplir la liste avec les valeurs des features présentes dans les données JSON
    features_data = []
    for feature_name in feature_names:
        if feature_name in data_dict:
            feature_value = data_dict[feature_name]
        else:
            feature_value = np.NaN
        features_data.append(feature_value)

    # Créer un DataFrame avec une seule ligne à partir des valeurs des features
    data = pd.DataFrame([features_data], columns=feature_names)

    # Prédiction :
    proba = model.predict_proba(data)[0, 1]
    prediction = (proba > THRESHOLD).astype(int)

    # Verbalisation de la prédiction :
    if prediction == 0:
        result = 'DOSSIER ACCEPTE'
    else:
        result = 'DOSSIER REFUSE'

    # Extraction du classifier du modèle :
    classifier = model.named_steps['classifier']

    # Création de l'explainer SHAP pour CatBoost :
    explainer = shap.TreeExplainer(classifier)

    # Calcul des valeurs Shap :
    shap_values = explainer(data)

    # Renvoi du résultat :
    return jsonify(    
        {'Dossier': result,
        'Probabilite': round(proba, 3),
        'Seuil': round(THRESHOLD, 3),
        'Shapvals': shap_values.values.tolist(),
        'Basevals': shap_values.base_values.tolist()}
    )


if __name__ == '__main__':
    app.run(debug=True)