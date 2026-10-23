#!/bin/bash -l
#SBATCH -o ../logs/validate_%j.out
#SBATCH -e ../logs/validate_%j.err
#SBATCH -J dynamics_plm_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Scores update_1000 ... update_20000 on the held-out proteins and writes the
# checkpoint with the lowest dynamics loss to save_dir/checkpoints/best.pt
conda activate dn_plm

PROJ_DIR="$(dirname "$SLURM_SUBMIT_DIR")"
cd "$PROJ_DIR/model"

python validate.py \
    --save_dir  "$PROJ_DIR/save_dir" \
    --max_len   1024 \
    --chk_start 1000 \
    --chk_end   20000 \
    --chk_step  1000
