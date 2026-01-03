import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

def label_encoder(df: pd.DataFrame) -> pd.DataFrame:

    """
    
    """

    df_copy = df.copy()
    columnsToEncode = list(df_copy.select_dtypes(include=['category', 'object'])) #TODO specify which columns need to be encoded? In the future, other object columns may need to be encoded differently? 
    #TODO decision tree model should be configured to treat h3_index as a categorical column?
    
    encoder_dict = {}
    for feature in columnsToEncode:
        try:
            le = LabelEncoder() #TODO Do we want to use LabelEncoder for encoding features? Also, creating Encode object for each feature (moved le object inside For Loop)
            df_copy[feature] = le.fit_transform(df_copy[feature])
            encoder_dict[feature] = le
        except Exception as e:
            logger.error(f'Error encoding {feature}: {e}') #TODO modified error handling here. Handles unseen values as well. 
            raise #TODO stopping execution
    
    return df_copy, encoder_dict

from sklearn.preprocessing import OrdinalEncoder

# def ordinal_encoder(df: pd.DataFrame, encoder = None) -> pd.DataFrame:

#     """

#     """
    
#     df_copy = df.copy()
#     columnsToEncode = list(df_copy.select_dtypes(include=['category', 'object']))
    
#     if not columnsToEncode:
#         return df_copy, None
    
#     try:
#         if encoder is None:
#             encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#             df_copy[columnsToEncode] = encoder.fit_transform(df_copy[columnsToEncode])
#         else:
#             df_copy[columnsToEncode] = encoder.transform(df_copy[columnsToEncode])
#     except Exception as e:
#         logger.error(f'Error encoding: {e}')
#         raise
    
#     return df_copy, encoder