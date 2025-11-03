from google.cloud import aiplatform
import os

# Initialize Vertex AI
PROJECT_ID = "dan-data-eng21-765b"
REGION = "us-central1"
BUCKET_NAME = "audio_swedish_2"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET_NAME}",
    service_account="project-service-account@dan-data-eng21-765b.iam.gserviceaccount.com",
)

# Use CustomContainerTrainingJob
job = aiplatform.CustomContainerTrainingJob(
    display_name="orpheus-tts-training",
    container_uri="us-central1-docker.pkg.dev/dan-data-eng21-765b/audio-jobs/audio-swedish-job_train:latest",
)

# Define the machine configuration
machine_type = "n1-standard-4"
accelerator_type = "NVIDIA_TESLA_T4"
accelerator_count = 1

# Submit the training job without model registration
job.run(
    replica_count=1,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    base_output_dir=f"gs://{BUCKET_NAME}/model-output",
)

print(f"Training job completed. Job name: {job.resource_name}")
