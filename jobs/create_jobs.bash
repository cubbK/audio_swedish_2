#!/bin/bash
# filepath: /Users/dan.popa/work/audio_swedish_2/run_jobs.sh

# Array of shard names
shards=(
    "dataset-000001"
    "dataset-000002"
    "dataset-000003"
    "dataset-000004"
    "dataset-000005"
    "dataset-000006"
    "dataset-000007"
    "dataset-000008"
    "dataset-000009"
    "dataset-000010"
    "dataset-000011"
)

# Loop through each shard and run the make command
for shard in "${shards[@]}"; do
    echo "Creating job for shard: $shard"
    JOB_NAME="$shard" make create_job_2
    
    # Optional: Add a small delay between jobs if needed
    sleep 1
done

echo "All jobs created successfully!"