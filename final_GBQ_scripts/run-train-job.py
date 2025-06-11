#Example script to run a custom job on Google Cloud Vertex AI
# This script initializes the Vertex AI environment and runs a custom job for training an XGBoost model.

from google.cloud import aiplatform

aiplatform.init(
    project="behavio-test",
    location="europe-west1",
    staging_bucket="gs://withdraws-model"
)

job = aiplatform.CustomJob(
    display_name="xgb-model-monthly-job",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "europe-docker.pkg.dev/behavio-test/vertex-containers/xgb-model:latest",
                "command": ["python3"],
                "args": ["/app/model_train_run_monthly.py"],
            },
        }
    ],
)

job.run(sync=True)