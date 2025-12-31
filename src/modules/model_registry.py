from google.cloud import aiplatform, storage
import joblib
import tempfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_object_in_registry(
    model_object,
    model_display_name: str,
    project_id: str,
    region: str,
    gcs_bucket: str,
    alias: str,
    description = None,
    labels = None,
    model_filename: str = "model.pkl",
    is_default_version: bool = False):
    
    """

    """

    logger.info("Saving object to Registry...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_model_path = f"models/{model_display_name}/{timestamp}"
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
        joblib.dump(model_object, tmp.name)
        blob = bucket.blob(f"{gcs_model_path}/{model_filename}")
        blob.upload_from_filename(tmp.name)
    
    gcs_model_uri = f"gs://{gcs_bucket}/{gcs_model_path}/{model_filename}"
    artifact_uri = f"gs://{gcs_bucket}/{gcs_model_path}"
    
    aiplatform.init(project=project_id, location=region)
    
    existing_models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    
    parent_model = existing_models[0].resource_name if existing_models else None
    vertex_model = aiplatform.Model.upload(
        display_name=model_display_name, 
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest", # plceholder to meet api requirements
        parent_model=parent_model,
        is_default_version=is_default_version,
        version_aliases=[alias],
        version_description=description,
        labels=labels
    )
    
    logger.info(f"Object saved to GCS: {gcs_model_uri}")
    logger.info(f"Object registered: {vertex_model.resource_name}")
    logger.info(f"Alias '{alias}' assigned")
    logger.info(f"Version ID: {vertex_model.version_id}")

def load_object_from_registry(
        display_name: str,
        project_id: str, 
        region: str, 
        gcs_bucket: str, 
        alias: str):
    
    """
    
    """

    logger.info("Loading object from Registry...")

    aiplatform.init(project=project_id, location=region)
    models = aiplatform.Model.list(filter=f'display_name="{display_name}"')
    
    if not models:
        raise ValueError(f"No object found with display name: {display_name}")
    
    model_id = models[0].name 
    
    vertex_model = aiplatform.Model(model_name=f"{model_id}@{alias}")
    
    logger.info(f"Loaded object: {vertex_model.display_name}")
    logger.info(f"Version ID: {vertex_model.version_id}")
    logger.info(f"Alias: {alias}")
    
    gcs_uri = vertex_model.uri
    storage_client = storage.Client(project=project_id)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        gcs_path = gcs_uri.replace("gs://", "")
        blob_path = "/".join(gcs_path.split("/")[1:])
        
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(f"{blob_path}/model.pkl")
        blob.download_to_filename(tmp.name)
        
        model_object = joblib.load(tmp.name)
    
    return model_object