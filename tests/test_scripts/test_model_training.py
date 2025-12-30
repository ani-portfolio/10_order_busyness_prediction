import os
os.environ['ENV'] = 'local'

import pandas as pd
import joblib
from src import config
from src.scripts import model_training


def test_model_training_pipeline(tmp_path):
    
    config.base = str(tmp_path)
    config.training_input_data = str(tmp_path / "training_input_data.csv")
    
    model_training.main()
    
    artifacts_path = tmp_path / "artifacts"
    
    assert os.path.exists(config.training_input_data), "Processed data not saved"
    assert os.path.exists(artifacts_path / "centroids.pkl"), "Centroids not saved"
    assert os.path.exists(artifacts_path / "encoder.pkl"), "Encoder not saved"
    assert os.path.exists(artifacts_path / "model.pkl"), "Model not saved"
    
    df_processed = pd.read_csv(config.training_input_data)
    encoder = joblib.load(artifacts_path / "encoder.pkl")
    centroids = joblib.load(artifacts_path / "centroids.pkl")
    model = joblib.load(artifacts_path / "model.pkl")
    
    assert not df_processed.empty, "Processed data is empty"
    assert all(col in df_processed.columns for col in config.feature_list), "Features missing"
    
    assert encoder is not None, "Encoder is None"
    assert centroids is not None, "Centroids is None"
    assert model is not None, "Model is None"