#!/bin/bash
#SBATCH --job-name=ngiab-job
#SBATCH --output=ngiab_%j.out
#SBATCH --error=ngiab_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --account=

if uname -a | grep arm64 || uname -a | grep aarch64 ; then
    ARCH=arm64
    IMAGE_URL=library://trupeshkumarpatel/awiciroh/ciroh-ngen-singularity:latest_arm
    IMAGE_NAME=$SCRATCH/ciroh-ngen-singularity_latest.sif
else
    ARCH=amd64
    IMAGE_URL=library://ciroh-it-support/ngiab/ciroh-ngen-singularity:latest_x86
    IMAGE_NAME=$SCRATCH/ciroh-ngen-singularity_latest.sif
fi


if [ ! -f "$IMAGE_NAME" ]; then
    echo "Pulling ngiab singularity image"
    singularity pull -F --arch "$ARCH" "$IMAGE_NAME" "$IMAGE_URL"
fi

# Default command to run inside the container (updated when creating job in Python)
CMD="/dmod/bin/ngen"

# Optional bind paths (host:container)
BIND_PATHS="$SCRATCH:/scratch"


echo "Hello, World from Anvil!"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Start time: $(date)"

# Execute the job
apptainer exec \
    --bind ${BIND_PATHS} \
    ${IMAGE_NAME} \
    ${CMD}

echo "End time: $(date)"
