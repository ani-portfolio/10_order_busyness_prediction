import os
import joblib
import logging
from src import config
from src.modules import model_registry

logger = logging.getLogger(__name__)

def save_artifact(obj, name, filename, alias):
    if config.ENV == 'local':
        local_path = f'{config.base}/artifacts/{filename}'
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        joblib.dump(obj, local_path)
        logger.info(f'Saved {name} locally to {local_path}')
    else:
        model_registry.save_object_in_registry(
            obj, name, config.gcp_project_id, config.gcp_region,
            config.gcp_bucket, alias, filename
        )
        logger.info(f'Saved {name} to Model Registry')

def load_artifact(name, filename, alias):
    if config.ENV == 'local':
        local_path = f'{config.base}/artifacts/{filename}'
        logger.info(f'Load {name} from {local_path}')
        return joblib.load(local_path)
    else:
        logger.info(f'Load {name} from Model Registry')
        return model_registry.load_object_from_registry(
            name, config.gcp_project_id, config.gcp_region,
            config.gcp_bucket, alias, filename
        )