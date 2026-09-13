import pytest
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_regression_sequence_0():
    response = client.get("/api/dataset/sequence/0")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "trajectory" in data
    
    # Check that predictions are consistent (P(Attack) shouldn't change for the same seq)
    # Sequence 0 is Benign with a specific trajectory
    assert data["current"]["pred_label"] == "Infiltration"
    assert "forecasts" in data
    assert len(data["forecasts"]) == 4

def test_regression_sequence_attack():
    # Sequence 1100 is typically an attack in this dataset (just pick a high index)
    # The exact index depends on the test split, we'll just test that it runs.
    response = client.get("/api/dataset/sequence/1000")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "is_attack" in data

if __name__ == "__main__":
    pytest.main([__file__])
