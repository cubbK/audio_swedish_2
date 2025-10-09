#!/bin/bash
# filepath: /Users/dan.popa/work/audio_swedish_2/run_jobs.sh

# Array of shard names
shards=(
    "jobs_dataset-000001"
    "jobs_dataset-000002"
    "jobs_dataset-000003"
    "jobs_dataset-000004"
    "jobs_dataset-000005"
    "jobs_dataset-000006"
    "jobs_dataset-000007"
    "jobs_dataset-000008"
    "jobs_dataset-000009"
    "jobs_dataset-000010"
    "jobs_dataset-000011"
)

# Loop through each shard and run the make command
for shard in "${shards[@]}"; do
    echo "Creating job for shard: $shard"
    JOB_NAME="$shard" make create_job_2
    
    # Optional: Add a small delay between jobs if needed
    sleep 1
done

echo "All jobs created successfully!"