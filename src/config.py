import os

################################################################ ENV CONFIG
ENV = os.getenv('ENV', 'local')  

gcp_project_id = os.getenv('GCP_PROJECT_ID', 'jet-tha')
gcp_region = os.getenv('GCP_REGION', 'us-central1')
gcp_artifact_repo = os.getenv('GCP_ARTIFACT_REPO', 'jet-docker')
gcp_bucket = os.getenv('GCP_BUCKET', 'jet-bucket-a')

BASE_PATH = {
    'local': '/Users/ani/Projects/JET_TAKE_HOME_ASSIGNMENT',
    'dev': 'gs://jet-bucket-a'
}

base = BASE_PATH.get(ENV, BASE_PATH['dev'])

################################################################ PATHS
input_data_path = f'{base}/data/input/training_data.csv'
inference_data_path = f'{base}/data/input/inference_data.csv'
training_input_data = f'{base}/data/processed/training_input_data.csv'
inference_input_data = f'{base}/data/processed/inference_input_data.csv'
predictions_path = f'{base}/data/output/predictions.csv'
centroids_path = f'{base}/artifacts/centroids.pkl'
encoder_path = f'{base}/artifacts/label_encoder.pkl'
trained_model_path = f'{base}/artifacts/trained_model.pkl'
logger_path = 'logs/' if ENV == 'local' else None

################################################################ PARAMETERS
granularity_id = ['h3_index', 'date_day_number', 'date_hour_number']

required_input_cols = ['courier_id',
 'order_number',
 'courier_location_timestamp',
 'courier_lat',
 'courier_lon',
 'order_created_timestamp',
 'restaurant_lat',
 'restaurant_lon']

k = 5
resolution = 7

target_variable = ['orders_busyness_by_h3_hour']

feature_list = [
    'dist_to_restaurant',
    'Hdist_to_restaurant',
    'avg_Hdist_to_restaurants',
    'date_day_number',
    'restaurant_id',
    'Five_Clusters_embedding',
    'h3_index',
    'date_hour_number',
    'restaurants_per_index'
]

cv_folds = 3
random_forest_params = {
    'max_depth': [4, 5],
    'min_samples_leaf': [50, 75],
    'n_estimators': [100, 150]
}

test_size = 0.33
split_random_state = 42
regressor_random_state = 0
regressor_n_jobs = -1
cv_n_jobs = -1
cv_scoring = 'r2'

################################################################ NULL RULES
null_rules = {
    'orders_busyness_by_h3_hour': 'no_nulls',
    'dist_to_restaurant': 'no_nulls',
    'Hdist_to_restaurant': 'no_nulls',
    'avg_Hdist_to_restaurants': {'max_pct': 0.05},
    'date_day_number': 'no_nulls',
    'restaurant_id': 'no_nulls',
    'Five_Clusters_embedding': 'no_nulls',
    'h3_index': 'no_nulls',
    'date_hour_number': 'no_nulls',
    'restaurants_per_index': 'no_nulls'
}

################################################################ SCHEMAS
input_data_schema = {
    'courier_id': 'object',
    'order_number': 'int64',
    'courier_location_timestamp': 'object',
    'courier_lat': 'float64',
    'courier_lon': 'float64',
    'order_created_timestamp': 'object',
    'restaurant_lat': 'float64',
    'restaurant_lon': 'float64'
}

processed_data_schema = {
    'courier_id': 'int64',
    'order_number': 'int64',
    'courier_location_timestamp': 'datetime64[ns, UTC]',
    'courier_lat': 'float64',
    'courier_lon': 'float64',
    'order_created_timestamp': 'datetime64[ns, UTC]',
    'restaurant_lat': 'float64',
    'restaurant_lon': 'float64',
    'restaurant_id': 'int64',
    'dist_to_restaurant': 'float64',
    'avg_dist_to_restaurants': 'float64',
    'Hdist_to_restaurant': 'float64',
    'avg_Hdist_to_restaurants': 'float64',
    'Five_Clusters_embedding': 'int64',
    'Five_Clusters_embedding_error': 'float64',
    'h3_index': 'int64',
    'date_day_number': 'int64',
    'date_hour_number': 'int64',
    'orders_busyness_by_h3_hour': 'int64',
    'restaurants_per_index': 'int64'
}

feature_schema = {
    'dist_to_restaurant': 'float64',
    'Hdist_to_restaurant': 'float64',
    'avg_Hdist_to_restaurants': 'float64',
    'date_day_number': 'int64',
    'restaurant_id': 'int64',
    'Five_Clusters_embedding': 'int64',
    'h3_index': 'int64',
    'date_hour_number': 'int64',
    'restaurants_per_index': 'int64'
}