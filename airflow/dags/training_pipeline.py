from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime
from config import IMAGE, IMAGE_PULL_POLICY, NAMESPACE, ENV, VOLUMES, VOLUME_MOUNTS

with DAG(
    'ml_training_pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    train_task = KubernetesPodOperator(
        task_id='train_model',
        name='training-pod',
        namespace=NAMESPACE,
        image_pull_policy=IMAGE_PULL_POLICY,
        image=IMAGE,
        cmds=['python', '-m', 'src.scripts.model_training'],
        env_vars={'ENV': ENV},
        volumes=VOLUMES,
        volume_mounts=VOLUME_MOUNTS,
        get_logs=True,
        is_delete_operator_pod=True,
    )