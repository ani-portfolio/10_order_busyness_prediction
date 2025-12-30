from sklearn.base import BaseEstimator
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_predictions(df: pd.DataFrame, features: list, granularity_id: list, model: BaseEstimator) -> pd.DataFrame:
    """
    
    """

    df_processed = df[features]

    try:
        preds = model.predict(df_processed)
        df_predictions = df[granularity_id]
        df_predictions['preds'] = preds
    
    except Exception as e:
        logger.error(f'Prediction failed for model: {e}')
        raise

    return df_predictions