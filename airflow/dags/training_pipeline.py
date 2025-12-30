import os
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime

PROJECT_ID = os.getenv('ML_PROJECT_ID', 'jet-tha')
REGION = os.getenv('ML_REGION', 'us-central1')
ARTIFACT_REPO = os.getenv('ML_ARTIFACT_REPO', 'jet-docker')

with DAG(
    'ml_training_pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    train_task = KubernetesPodOperator(
        task_id='train_model',
        name='training-pod',
        namespace='default',
        image=f'{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/training:latest',
        cmds=['python', '-m', 'src.scripts.model_training'],
        env_vars={'ENV': os.getenv('ENV', 'dev')},
        get_logs=True,
        is_delete_operator_pod=True,
    )