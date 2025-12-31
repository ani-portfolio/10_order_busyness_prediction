import pandas as pd
from fastapi import FastAPI
import logging

from src.modules import data_validation, predict
from src.scripts import data_processing
from src import config
from src.utils import route

logger = logging.getLogger(__name__)

app = FastAPI()

encoder = route.load_artifact('label_encoder', 'encoder.pkl', 'production')
model = route.load_artifact('trained_model', 'model.pkl', 'production')


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_endpoint(request: dict):
    instances = request.get("instances", [])
    df_inference_data = pd.DataFrame(instances)

    data_validation.validate_schema(df_inference_data, config.input_data_schema)

    df_processed = data_processing.process_data(df_inference_data, mode='inference')

    for feature, enc in encoder.items():
        df_processed[feature] = enc.transform(df_processed[feature])

    data_validation.validate_schema(df_processed, config.processed_data_schema)
    data_validation.validate_nulls(df_processed, config.null_rules)

    df_predictions = predict.get_predictions(
        df_processed,
        config.feature_list,
        config.granularity_id,
        model
    )

    return {"predictions": df_predictions.to_dict(orient='records')}