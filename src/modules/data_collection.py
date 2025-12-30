# from io import StringIO
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def import_raw_data(path: str) -> pd.DataFrame:

    """
    
    """

    df = None
    try:
        df = pd.read_csv(path)

        df.dropna(axis=0, inplace=True) # TODO discuss with DS. Maybe dropna should happen in a separate step outside data loading?
    
    except Exception as e:
        logger.info(f'Error loading data: {e}')
        raise

    return df
