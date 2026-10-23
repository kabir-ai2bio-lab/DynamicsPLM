#!/bin/bash -l
#SBATCH -o ../logs/eval_proteingym_%j.out
#SBATCH -e ../logs/eval_proteingym_%j.err
#SBATCH -J dynamics_plm_proteingym
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Resumable: re-submitting skips proteins already in summary.csv
conda activate dn_plm

PROJ_DIR="$(dirname "$SLURM_SUBMIT_DIR")"
MODEL="dynamicsplm"                                   # dynamicsplm | esm2 | esmdance
CHECKPOINT="$PROJ_DIR/save_dir/checkpoints/best.pt"   # used for dynamicsplm only

CKPT_ARG=""
[ "$MODEL" = "dynamicsplm" ] && CKPT_ARG="--checkpoint $CHECKPOINT"

cd "$PROJ_DIR"

python evaluation/proteingym.py \
    --model      "$MODEL" \
    --save_dir   "$PROJ_DIR/save_dir" \
    $CKPT_ARG
