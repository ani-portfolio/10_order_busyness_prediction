import pandas as pd
from src.modules.encode import label_encoder


def test_label_encoder():
    
    df_test = pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C'],
        'object_col': ['x', 'y', 'x', 'z'],
        'numeric_col': [1, 2, 3, 4]
    })
    
    df_result, encoder_dict = label_encoder(df_test)
    
    assert 'category_col' in encoder_dict, "category_col not in encoder_dict"
    assert 'object_col' in encoder_dict, "object_col not in encoder_dict"
    assert 'numeric_col' not in encoder_dict, "numeric_col should not be encoded"
    assert len(encoder_dict) == 2, "Number of encoders is incorrect"
    assert df_result['category_col'].dtype in ['int32', 'int64'], "category_col not encoded"
    assert df_result['object_col'].dtype in ['int32', 'int64'], "object_col not encoded"
