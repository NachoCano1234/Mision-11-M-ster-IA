from fastapi.testclient import TestClient
from src.api import app

def test_health():
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_invalid_features():
    response = TestClient(app).post('/predict', json={'features': [1.0]})
    assert response.status_code == 422
