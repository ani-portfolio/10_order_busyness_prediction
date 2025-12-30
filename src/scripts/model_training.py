import os
import joblib
from datetime import timedelta
import tracemalloc
import time

from src.modules import data_collection, encode, train, data_validation, model_registry
from src.scripts import data_processing
from src.utils import logging
import logging
from src import config

logger = logging.getLogger(__name__)

def main():

    start_time = time.time()

    logger.info('Load raw data')
    df_raw_data = data_collection.import_raw_data(config.input_data_path)

    data_validation.validate_schema(df_raw_data, config.input_data_schema)

    logger.info('Process raw data')
    df_processed = data_processing.process_data(df_raw_data, mode='training')

    logger.info('Encode data')
    df_processed, encoder = encode.label_encoder(df_processed)

    logger.info('Save training input data')
    df_processed.to_csv(config.training_input_data)

    data_validation.validate_schema(df_processed, config.processed_data_schema)
    data_validation.validate_nulls(df_processed, config.null_rules)

    logger.info('Save encoder')
    model_registry.save_object_in_registry(encoder, 'label_encoder', config.gcp_project_id, config.gcp_region, config.gcp_bucket, 'production', "encoder.pkl")

    logger.info('Train-Test split')
    X_train, X_test, y_train, y_test = train.split_data(df_processed, config.feature_list, config.target_variable, test_size=config.test_size, random_state=config.split_random_state)

    data_validation.validate_schema(X_train, config.feature_schema)
    data_validation.validate_schema(X_test, config.feature_schema)

    logger.info('Hyper-parameter tuning')
    grid_search = train.hyperparameter_tuning(X_train, y_train, config.random_forest_params, config.cv_folds, config.cv_scoring, config.regressor_n_jobs, config.cv_n_jobs, config.regressor_random_state)

    best_model = grid_search.best_estimator_
    logger.info(f'Test score {config.cv_scoring} {best_model.score(X_test, y_test)}')

    logger.info('Refit model on all data')
    best_params = grid_search.best_params_
    refitted_model = train.train_refitted_model(X_train, y_train, X_test, y_test, best_params)

    logger.info('Save refitted model')
    model_registry.save_object_in_registry(refitted_model, 'trained_model', config.gcp_project_id, config.gcp_region, config.gcp_bucket, 'production', "model.pkl")

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {str(timedelta(seconds=training_time))}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
