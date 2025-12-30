import pandas as pd
from src.modules.feature_engineering import (
    get_restaurant_ids
)


def test_get_restaurant_ids():

    df_test = pd.DataFrame({
        'restaurant_lat': [40.7, 40.7, 40.8],
        'restaurant_lon': [-74.0, -74.0, -74.1]
    })
    
    restaurants_ids, df_result = get_restaurant_ids(df_test)
    
    assert len(restaurants_ids) == 2, "Number of restaurant ids is incorrect"
    assert 'restaurant_id' in df_result.columns, "restaurant_id column not in df"
    assert df_result['restaurant_id'].nunique() == 2, "Number of unique restaurant_id is incorrect"

