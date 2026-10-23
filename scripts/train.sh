#!/bin/bash -l
#SBATCH -o ../logs/train_%j.out
#SBATCH -e ../logs/train_%j.err
#SBATCH -J dynamics_plm_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Submit from scripts/:  cd scripts && sbatch train.sh
conda activate dn_plm

PROJ_DIR="$(dirname "$SLURM_SUBMIT_DIR")"
SAVE_DIR="$PROJ_DIR/save_dir"

# Loss weights of Eq. (1). Published model: 0.7 / 0.3.
LAMBDA_MLM=0.7
LAMBDA_DYNAMICS=0.3

if [ -f "$SAVE_DIR/checkpoints/latest.pt" ]; then
    echo "Found existing checkpoint - RESUMING training"
    RESUME_ARG="--resume latest"
else
    echo "No checkpoint found - STARTING fresh training"
    RESUME_ARG=""
fi

cd "$PROJ_DIR/model"

python -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py \
    --save_dir "$SAVE_DIR" \
    --lambda_mlm "$LAMBDA_MLM" \
    --lambda_dynamics "$LAMBDA_DYNAMICS" \
    $RESUME_ARG
