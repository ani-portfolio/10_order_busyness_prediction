import pandas as pd
from src import config
from src.modules import feature_engineering, model_registry
import logging

logger = logging.getLogger(__name__)

def process_data(df_raw_data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """

    """

    logger.info('Data processing started...')
    df_raw_courier_data = df_raw_data.copy()

    restaurants_ids, df_processed = feature_engineering.get_restaurant_ids(df_raw_courier_data)

    df_processed['dist_to_restaurant'] = feature_engineering.calc_dist(df_processed.courier_lat, df_processed.courier_lon, df_processed.restaurant_lat, df_processed.restaurant_lon)

    df_processed['avg_dist_to_restaurants'] = [feature_engineering.avg_dist_to_restaurants(lat,lon,restaurants_ids) for lat,lon in zip(df_processed.courier_lat, df_processed.courier_lon)]

    df_processed['Hdist_to_restaurant'] = feature_engineering.calc_haversine_dist(df_processed.courier_lat.tolist(), df_processed.courier_lon.tolist(), df_processed.restaurant_lat.tolist(), df_processed.restaurant_lon.tolist())

    df_processed['avg_Hdist_to_restaurants'] = [feature_engineering.avg_Hdist_to_restaurants(lat,lon,restaurants_ids) for lat,lon in zip(df_processed.courier_lat, df_processed.courier_lon)]

    if mode == 'training':

        df_restaurants = pd.DataFrame([{"lat": v['lat'], "lon": v['lon']} for v in restaurants_ids.values()]) #TODO Add this to get_restaurant_ids function?

        centroids = feature_engineering.initiate_centroids(config.k, df_restaurants) #TODO Should this only happen during model training? The same centroids should then be used in inference

        logger.info('Save centroids')
        model_registry.save_object_in_registry(centroids, 'centroids', config.gcp_project_id, config.gcp_region, config.gcp_bucket, 'production', "centroids.pkl")
    
    elif mode == 'inference':
        
        logger.info('Load centroids')
        centroids = model_registry.load_object_from_registry("centroids", "production", config.gcp_project_id, config.gcp_region)

    df_processed = feature_engineering.centroid_assignation(df_processed, centroids)

    df_processed = feature_engineering.create_h3_index(df_processed, config.resolution)

    df_processed = feature_engineering.date_time_features(df_processed)

    df_processed = feature_engineering.orders_busyness(df_processed)

    df_processed = feature_engineering.restaurant_index(df_processed)

    logger.info('Data processing complete...')

    return df_processed
