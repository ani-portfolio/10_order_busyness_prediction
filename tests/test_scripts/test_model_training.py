import pandas as pd
import joblib
import os
from src.scripts import model_training
from src import config


def test_model_training_pipeline(tmp_path):
    
    # Setup - override config paths to use temp directory
    config.raw_data_path = "/Users/ani/Projects/skip_coding_interview/data/pytest/data/raw_data_test.csv"
    config.processed_data_path = tmp_path / "processed_data.csv"
    config.encoder_path = tmp_path / "encoder.pkl"
    config.trained_model_path = tmp_path / "model.pkl"
    
    # Run pipeline
    model_training.main()
    
    # Check outputs exist
    assert os.path.exists(config.processed_data_path), "Processed data not saved"
    assert os.path.exists(config.encoder_path), "Encoder not saved"
    assert os.path.exists(config.trained_model_path), "Model not saved"
    
    # Load and validate outputs
    df_processed = pd.read_csv(config.processed_data_path)
    encoder = joblib.load(config.encoder_path)
    model = joblib.load(config.trained_model_path)
    
    # Check processed data
    assert not df_processed.empty, "Processed data is empty"
    assert all(col in df_processed.columns for col in config.feature_list), "Features missing"
    
    # Check encoder
    assert encoder is not None, "Encoder is None"
    