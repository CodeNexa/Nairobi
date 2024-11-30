import pytest
import pandas as pd
from src.integrity_analysis import preprocess_data, calculate_scores, IntegrityModel

# Sample dataset for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Platform": ["Bolt", "Faras", "Uber"],
        "Honesty": [80, 60, 70],
        "Transparency": [70, 50, 65],
        "Accountability": [60, 40, 55],
        "Ethics": [75, 55, 65],
        "Consistency": [85, 65, 75]
    })

def test_preprocess_data(sample_data):
    """
    Test that data preprocessing handles missing values and scales correctly.
    """
    processed_data = preprocess_data(sample_data)
    assert "Platform" in processed_data.columns
    assert all(processed_data.drop(columns=["Platform"]).max() <= 1.0)  # Values should be scaled between 0-1
    assert all(processed_data.drop(columns=["Platform"]).min() >= 0.0)

def test_calculate_scores(sample_data):
    """
    Test that the integrity scoring function calculates scores accurately.
    """
    scores = calculate_scores(sample_data)
    assert len(scores) == len(sample_data)  # Scores should match the number of rows
    assert all(isinstance(score, float) for score in scores)  # All scores should be floats

def test_model_training(sample_data):
    """
    Test the model training process.
    """
    model = IntegrityModel()
    model.train(sample_data)
    assert model.is_trained  # Ensure the model is flagged as trained

def test_model_prediction(sample_data):
    """
    Test that the model makes predictions correctly.
    """
    model = IntegrityModel()
    model.train(sample_data)  # Train the model first
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)  # Predictions should match the number of rows
    assert all(isinstance(pred, float) for pred in predictions)  # All predictions should be floats
