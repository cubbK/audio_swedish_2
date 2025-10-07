PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1  # or your region

SERVICE_ACCOUNT=$(gcloud projects describe $PROJECT_ID \
  --format="value(projectNumber)")-compute@developer.gserviceaccount.com

gcloud artifacts repositories add-iam-policy-binding YOUR_REPO \
  --location=$REGION \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/artifactregistry.reader"