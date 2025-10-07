JOB_NAME ?= dataset-000000
PROJECT_ID ?= dan-data-eng-20-ce03
REGION ?= europe-west4
REPO_NAME ?= audio-jobs
IMAGE_NAME ?= audio-swedish-job
IMAGE_TAG ?= latest
IMAGE_URI = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):$(IMAGE_TAG)

# Create Artifact Registry repository (run once)
create_repo:
	gcloud artifacts repositories create $(REPO_NAME) \
    --repository-format=docker \
    --location=$(REGION) \
    --description="Audio processing jobs"

# Configure Docker authentication
configure_docker:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

build_image:
	docker build -t $(IMAGE_URI) .

push_image: build_image
	docker push $(IMAGE_URI)

# create_job:
# 	gcloud ai custom-jobs create \
#     --region=$(REGION) \
#     --display-name=$(JOB_NAME) \
#     --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,executor-image-uri=$(IMAGE_URI),local-package-path=.,script=job2.py

create_job_2:
	gcloud ai custom-jobs create \
	--region=$(REGION) \
	--display-name=$(JOB_NAME) \
	--worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=$(IMAGE_URI)

