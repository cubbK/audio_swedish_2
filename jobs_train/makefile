JOB_NAME ?= train_simple
PROJECT_ID ?= dan-data-eng21-765b
REGION ?= us-central1
REPO_NAME ?= audio-jobs
IMAGE_NAME ?= audio-swedish-job_train
IMAGE_TAG ?= latest
IMAGE_URI = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):$(IMAGE_TAG)

# Configure Docker authentication
configure_docker:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

build_image:
	docker build -t $(IMAGE_URI) .

push_image: build_image
	docker push $(IMAGE_URI)

create_job_2:
	gcloud ai custom-jobs create \
	--region=$(REGION) \
	--display-name=$(JOB_NAME) \
	--worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=$(IMAGE_URI) \
	--args="--dataset-shard,$(JOB_NAME)"