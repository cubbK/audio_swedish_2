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
# machine_type = "n1-standard-4"
# accelerator_type = "NVIDIA_TESLA_T4"
# accelerator_count = 1

machine_type = "a2-highgpu-1g"
accelerator_type = "NVIDIA_TESLA_A100"
accelerator_count = 1

HF_TOKEN = os.getenv("HF_TOKEN", "your-huggingface-token-here")

print(HF_TOKEN)

# Submit the training job without model registration
job.run(
    replica_count=1,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    base_output_dir=f"gs://{BUCKET_NAME}/model-output",
    environment_variables={
        "HF_TOKEN": HF_TOKEN,
    },
    boot_disk_size_gb=200,
    sync=False,  # set to true to wait for job completion
)

print(f"Training job completed. Job name: {job.resource_name}")

print(f"Monitor the job at: {job._dashboard_uri()}")
