import joblib
from datetime import timedelta
import tracemalloc
import time

from src.modules import data_collection, predict, data_validation, model_registry
from src.scripts import data_processing
from src.utils import logging
import logging
from src import config
from src.utils import route

logger = logging.getLogger(__name__)

def main():

    start_time = time.time()

    logger.info('Load raw data')
    df_inference_data = data_collection.import_raw_data(config.inference_data_path)

    data_validation.validate_schema(df_inference_data, config.input_data_schema)

    logger.info('Process raw data')
    df_processed = data_processing.process_data(df_inference_data, mode='inference')
    
    logger.info('Load encoder & model')
    encoder = route.load_artifact('label_encoder', 'encoder.pkl', 'production')
    model = route.load_artifact('trained_model', 'model.pkl', 'production')

    logger.info('Encode Features')
    for feature, enc in encoder.items():
        df_processed[feature] = enc.transform(df_processed[feature])

    data_validation.validate_schema(df_processed, config.processed_data_schema)
    data_validation.validate_nulls(df_processed, config.null_rules)

    logger.info('Save inference input data')
    df_processed.to_csv(config.inference_input_data)
    
    logger.info('Predict')

    df_predictions = predict.get_predictions(df_processed, config.feature_list, config.granularity_id, model)

    logger.info('Save predictions')
    df_predictions.to_csv(config.predictions_path) #TODO change where output is saved

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Inference completed {str(timedelta(seconds=training_time))}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()