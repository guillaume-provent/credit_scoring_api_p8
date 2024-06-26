import pytest
import joblib
from api import app, model_path


# Configuration de pytest pour utiliser le client de test de Flask :
@pytest.fixture
def client():

    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# Test de chargement du modèle :
def test_model_loading():

    model = joblib.load(model_path)
    assert model is not None


# Test de la prédiction :
def test_predict_valid_id(client):

    data = {'AMT_GOODS_PRICE': 100, 'AMT_ANNUITY': 20, 'AMT_CREDIT': 100}
    response = client.post('/predict', json=data)
    json_data = response.get_json()

    assert response.status_code == 200
    assert 'Dossier' in json_data
    assert 'Probabilite' in json_data
    assert 'Seuil' in json_data
    assert 'Shapvals' in json_data
    assert 'Basevals' in json_data
