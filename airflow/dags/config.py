import os
from kubernetes.client import V1Volume, V1VolumeMount, V1HostPathVolumeSource

ENV = os.getenv('ENV', 'local')

if ENV == 'local':
    IMAGE = 'ml-training:latest'
    IMAGE_PULL_POLICY = 'Never'
    NAMESPACE = 'default'
    PROJECT_PATH = '/Users/ani/Projects/JET_TAKE_HOME_ASSIGNMENT'
    VOLUMES = [
        V1Volume(
            name='project-data',
            host_path=V1HostPathVolumeSource(path=PROJECT_PATH)
        )
    ]
    VOLUME_MOUNTS = [
        V1VolumeMount(
            name='project-data',
            mount_path=PROJECT_PATH
        )
    ]
else:
    PROJECT_ID = os.getenv('ML_PROJECT_ID', 'jet-tha')
    REGION = os.getenv('ML_REGION', 'us-central1')
    ARTIFACT_REPO = os.getenv('ML_ARTIFACT_REPO', 'jet-docker')
    IMAGE = f'{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/training:latest'
    IMAGE_PULL_POLICY = 'Always'
    NAMESPACE = 'default'
    VOLUMES = []
    VOLUME_MOUNTS = []