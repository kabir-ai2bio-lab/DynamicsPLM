# DynamicsPLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23602205.svg)](https://doi.org/10.5281/zenodo.23602205)

Code and results for **"DynamicsPLM: A Dynamics-Enriched Small Protein Language Model for Protein Variant Effect Prediction"** (Dao, Shehu, Kabir, 2026).

DynamicsPLM continues pretraining ESM2-35M with three jointly trained heads: masked language modelling (MLM), residue-level dynamics (50 features) and pairwise dynamics (13 features). Training uses molecular dynamics data from ~62k proteins. At inference it scores mutations zero-shot from sequence alone, as the evolutionary (MLM) score minus the dynamics disruption score.

```
DynamicsPLM/
├── model/
│   ├── config.py            # hyperparameters, feature indices, data paths, loss weights
│   ├── model.py             # DynamicsPLM (ESM2 encoder + MLM / residue / pairwise heads)
│   ├── dataset.py           # BERT masking, cropping, three source-group dataloaders
│   ├── schedule.py          # warmup → linear decay → floor LR schedule
│   ├── train.py             # second-stage multi-task pretraining (DDP)
│   └── validate.py          # held-out dynamics loss per checkpoint → checkpoints/best.pt
├── evaluation/
│   ├── scoring.py           # zero-shot scores: MLM (Eq. 3), dynamics (Eq. 4), combined (Eq. 5)
│   ├── proteingym.py        # ProteinGym DMS substitution benchmark
│   ├── tsuboyama.py         # Tsuboyama 2023 stability benchmark (ddG / dG)
│   ├── dataset_statistics.py# training-set counts (Section III-A, Table S1)
│   └── util.py              # model loading, paths
├── scripts/                 # SLURM launchers: train, validate, eval_proteingym, eval_tsuboyama
├── dataset/                 # downloaded data (not tracked)
├── logs/                    # SLURM logs
└── save_dir/                # training and evaluation outputs (created when you run the code)
    ├── checkpoints/         # put the released weights here (Zenodo)
    └── evaluation/          # zero-shot results
```

## 1. Environment

```bash
conda create -n dn_plm python=3.12 -y
conda activate dn_plm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The ESMDance baseline also needs the SeqDance source code:

```bash
git clone https://github.com/ShenLab/SeqDance evaluation/SeqDance
```

## 2. Data

| Data | Source | Location |
|---|---|---|
| Training data (MD / IDRome / Proteinflow features) | [ChaoHou/protein_dynamic_properties](https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties) | `dataset/protein_dynamic_properties/` |
| ProteinGym DMS substitutions + `DMS_substitutions.csv` | [proteingym.org](https://proteingym.org/) | `dataset/DMS_ProteinGym_substitutions/` |
| Tsuboyama 2023 (`Tsuboyama2023_Dataset2_Dataset3_20230416.csv`) | [Zenodo 7992926](https://zenodo.org/records/7992926) | `dataset/` |

```python
from huggingface_hub import snapshot_download
snapshot_download("ChaoHou/protein_dynamic_properties", repo_type="dataset",
                  local_dir="dataset/protein_dynamic_properties")
```

The training folder must contain `training_test_data_sequence.csv`, `feature_weight.csv` and the three `feature_*.h5` files listed in `model/config.py`. To use another location, pass `--data_dir`.

## 3. Pretrained weights

The trained weights and the paper's zero-shot results are on Zenodo: [10.5281/zenodo.23602205](https://doi.org/10.5281/zenodo.23602205).

| File | Content |
|---|---|
| `dynamicsplm_35M.pt` | DynamicsPLM weights (λ_MLM = 0.7, λ_dyn = 0.3) |
| `results_proteingym_dynamicsplm.zip` | ProteinGym results: per-assay Spearman ρ and per-variant scores |
| `results_tsuboyama_ddG_dynamicsplm.zip` | Tsuboyama ΔΔG results |
| `results_tsuboyama_dG_dynamicsplm.zip` | Tsuboyama ΔG results |

Download `dynamicsplm_35M.pt` and save it as `save_dir/checkpoints/best.pt`. All evaluation scripts load this file by default:

```bash
wget -O save_dir/checkpoints/best.pt \
  "https://zenodo.org/records/23602205/files/dynamicsplm_35M.pt?download=1"
```

The result archives contain the original output files of the paper's evaluation runs. Their column names differ from the outputs of the scripts in this repository: `rho_zscore` (ProteinGym) and `rho_combined` (Tsuboyama) are the combined score, and `rho_dyn` is `rho_dynamics`.

## 4. Training

```bash
cd scripts && sbatch train.sh          # or, from model/:
torchrun --nproc_per_node=<N_GPUS> train.py --save_dir ../save_dir
```

The defaults reproduce the published model: 20k updates (15k at length ≤256, then 5k at ≤1024), effective batch size 32, and the encoder frozen for the first 2 dataloader epochs (an "epoch" here is one dataloader refresh, every 200 updates). Pass `--resume latest` to continue an interrupted run.

**Changing the MLM / dynamics loss ratio.** The published model uses λ_MLM = 0.7 and λ_dyn = 0.3 (Eq. 1). The paper also compares 0.5/0.5 and 0.3/0.7 (Section V-C). To change the ratio, do one of the following:

- edit `LAMBDA_MLM` / `LAMBDA_DYNAMICS` in `scripts/train.sh`;
- pass `--lambda_mlm 0.5 --lambda_dynamics 0.5` to `train.py`;
- change the defaults `lambda_mlm` / `lambda_dynamics` in `model/config.py`.

Use a separate `--save_dir` for each ratio so the checkpoints don't overwrite each other.

**Checkpoint selection.** Run `validate.py` (or `sbatch validate.sh`). It scores `update_1000.pt` … `update_20000.pt` on the 3,127 held-out proteins and copies the checkpoint with the lowest dynamics loss to `checkpoints/best.pt`:

```bash
cd model && python validate.py --save_dir ../save_dir
```

Training writes to `save_dir/`:

| File | Content |
|---|---|
| `checkpoints/update_{N}.pt` | snapshot every 1,000 updates (model, optimizer and RNG state) |
| `checkpoints/latest.pt`, `final_model.pt` | most recent / end-of-training checkpoint |
| `checkpoints/best.pt` | weights selected by `validate.py` |
| `training_config.json`, `training_log.json` | run settings; losses and learning rates every 20 updates |
| `validation_maxlen_1024.json` | held-out MLM and dynamics losses per checkpoint |

## 5. Zero-shot evaluation

Run these from the project root, or submit `scripts/eval_proteingym.sh` / `scripts/eval_tsuboyama.sh` (set `MODEL` and `LABEL` at the top of the script). Each run is resumable: proteins already in `summary.csv` are skipped.

```bash
# ProteinGym (201 assays ≤1022 aa)
python evaluation/proteingym.py --model dynamicsplm     # MLM + dynamics + combined
python evaluation/proteingym.py --model esmdance        # ESMDance baseline (dynamics only)
python evaluation/proteingym.py --model esm2            # ESM2-35M baseline (MLM only)

# Tsuboyama (412 proteins for ddG)
python evaluation/tsuboyama.py --model dynamicsplm --label ddG
python evaluation/tsuboyama.py --model dynamicsplm --label dG
python evaluation/tsuboyama.py --model esmdance    --label ddG
```

Use `--protein_id <DMS_id>` (ProteinGym) or `--protein <WT_name>` (Tsuboyama) to score a single protein, and `--checkpoint` to choose other weights.

Each run writes to `save_dir/evaluation/{proteingym|tsuboyama_ddG|tsuboyama_dG}/{model}/`:

| File | Content |
|---|---|
| `summary.csv` | one row per protein: `n_variants`, `rho_mlm`, `rho_dynamics`, `rho_combined` (Spearman ρ against the experimental score) |
| `per_protein/<name>.csv` | one row per variant: `mutation`, `mutated_sequence`, `dms_score`, `mlm_score`, `dynamics_score`, `combined_score` |

A higher dynamics score means more disruption, so `rho_dynamics` is negative; the paper reports its absolute value. The script prints the mean and median ρ over all proteins when it finishes.

To reproduce the paper's ablation (Section V-C), train a model for each ratio, select its `best.pt`, and run both benchmarks with `--checkpoint <that best.pt> --save_dir <its own save_dir>`.

## 6. Dataset statistics

```bash
python evaluation/dataset_statistics.py --split all
```

This prints the protein counts per source and split (Section III-A, Table S1) and saves feature-distribution figures to `save_dir/figures/`.

## Citation

```bibtex
@misc{dao2026dynamicsplm,
  title  = {DynamicsPLM: A Dynamics-Enriched Small Protein Language Model for Protein Variant Effect Prediction},
  author = {Dao, Tuan Minh and Shehu, Amarda and Kabir, Anowarul},
  year   = {2026}
}
```
