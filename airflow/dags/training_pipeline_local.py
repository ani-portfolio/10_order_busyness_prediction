import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from docker.types import Mount

PROJECT_ID = os.getenv('ML_PROJECT_ID', 'jet-tha')
REGION = os.getenv('ML_REGION', 'us-central1')
ARTIFACT_REPO = os.getenv('ML_ARTIFACT_REPO', 'jet-docker')

with DAG(
    'ml_training_pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    # LOCAL TESTING
    train_task = DockerOperator(
        task_id='train_model',
        image='ml-training:latest',
        command='python -m src.scripts.model_training',
        network_mode='bridge',
        auto_remove='success',
        mounts=[
            Mount(
                source='/Users/ani/Projects/skip_coding_interview',
                target='/Users/ani/Projects/skip_coding_interview',
                type='bind'
            )
        ],
        environment={'ENV': 'local'},
    )
