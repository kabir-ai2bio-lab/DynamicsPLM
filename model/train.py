import os
import time
import pandas as pd
import random
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from config import config
from model import DynamicsPLM
from schedule import WarmupDecaySchedule
from dataset import get_dataloader_cycle_iter

torch.manual_seed(config['training']['random_seed'])
torch.cuda.manual_seed_all(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])
random.seed(config['training']['random_seed'])

MODEL_SELECT = 'dynamics_plm'
ESM2_SELECT  = 'model_35M'

mseloss     = nn.MSELoss()
loss_weight = None   # per-feature / per-source weight table w_{f,s}, set by load_loss_weight()


def load_loss_weight(data_dir):
    global loss_weight
    loss_weight = pd.read_csv(
        os.path.join(data_dir, config['file_path']['loss_weight']), index_col=0)
    return loss_weight


def get_h5py_paths(data_dir):
    paths = [os.path.join(data_dir, name) for name in config['file_path']['h5_files']]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
    return paths


def calculate_mlm_loss(mlm_logits, original_input_ids, mask_positions, attention_mask):
    valid_positions = mask_positions & (attention_mask == 1)

    if valid_positions.sum() == 0:
        return torch.tensor(0.0, device=mlm_logits.device), 0, 0

    logits_flat  = mlm_logits.view(-1, mlm_logits.size(-1))
    targets_flat = original_input_ids.view(-1)
    valid_flat   = valid_positions.view(-1)

    loss        = F.cross_entropy(logits_flat[valid_flat], targets_flat[valid_flat])
    predictions = torch.argmax(logits_flat[valid_flat], dim=-1)
    correct     = (predictions == targets_flat[valid_flat]).sum().item()
    total       = targets_flat[valid_flat].numel()

    return loss, correct, total


def _per_protein_rmse(pred, target, valid_mask):
    """Mean over proteins of the RMSE over valid (non-sentinel) entries (SI Eq. 1-2)."""
    return torch.stack([
        torch.sqrt(mseloss(pred[p][valid_mask[p]], target[p][valid_mask[p]]))
        for p in range(len(target))
    ]).mean()


def calculate_dynamics_loss(source, output, res_feat, pair_feat):
    losses = {}
    res_idx  = config['training']['res_feature_idx']
    pair_idx = config['training']['pair_feature_idx']

    valid_res_mask  = (res_feat[:, :, 0] != -1)
    valid_pair_mask = (pair_feat[:, :, :, 0] != -1)

    if source in ['ATLAS_GPCRmd_PED_mdCATH', 'IDRome']:
        for feature in res_idx:
            if 'nma' not in feature:
                losses[feature] = _per_protein_rmse(
                    output[feature], res_feat[:, :, res_idx[feature]], valid_res_mask)
        for feature in pair_idx:
            if 'nma' not in feature:
                losses[feature] = _per_protein_rmse(
                    output[feature], pair_feat[:, :, :, pair_idx[feature]], valid_pair_mask)

    elif source in ['Proteinflow']:
        # Proteinflow HDF5 entries only hold the 3 NMA modes (columns 0-2)
        for k in range(3):
            losses[f'nma_res{k+1}'] = _per_protein_rmse(
                output[f'nma_res{k+1}'], res_feat[:, :, k], valid_res_mask)
            losses[f'nma_pair{k+1}'] = _per_protein_rmse(
                output[f'nma_pair{k+1}'], pair_feat[:, :, :, k], valid_pair_mask)

    weighted_losses = [
        losses[feature] * loss_weight.loc[feature, source]
        for feature in losses
    ]
    losses['dynamics_loss'] = torch.stack(weighted_losses).sum()
    return {f'{source}_{key}': losses[key] for key in losses}


def calculate_total_loss(source, output, batch, lambda_mlm, lambda_dynamics, num_sources=1):
    mlm_loss, mlm_correct, mlm_total = calculate_mlm_loss(
        output['mlm_logits'],
        batch['original_input_ids'],
        batch['mask_positions'],
        batch['attention_mask'],
    )
    all_losses = {
        'mlm_loss':    mlm_loss / num_sources,
        'mlm_correct': mlm_correct,
        'mlm_total':   mlm_total,
    }

    dynamics_losses = calculate_dynamics_loss(
        source, output, batch['res_feat'], batch['pair_feat'])
    all_losses.update(dynamics_losses)

    all_losses['total_loss'] = (
        lambda_mlm * (mlm_loss / num_sources)
        + lambda_dynamics * dynamics_losses[f'{source}_dynamics_loss']
    )
    return all_losses


def save_checkpoint(save_dir, ddp_model, optimizer, scheduler, scaler,
                    n_update, n_epoch, n_batch_in_epoch, max_len, rank, is_final=False):
    if rank != 0:
        return

    checkpoint = {
        'model_state_dict':     ddp_model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'n_update':             n_update,
        'n_epoch':              n_epoch,
        'n_batch_in_epoch':     n_batch_in_epoch,
        'max_len':              max_len,
        'torch_rng_state':      torch.get_rng_state(),
        'torch_cuda_rng_state': torch.cuda.get_rng_state_all(),
        'numpy_rng_state':      np.random.get_state(),
        'python_rng_state':     random.getstate(),
    }

    chk_path = os.path.join(
        save_dir, 'checkpoints',
        'final_model.pt' if is_final else f'update_{n_update}.pt'
    )
    torch.save(checkpoint, chk_path)
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', 'latest.pt'))

    print(f"Checkpoint saved: {chk_path}  (update={n_update}, epoch={n_epoch})")
    return chk_path


def load_checkpoint(checkpoint_path, ddp_model, optimizer, scheduler, scaler, rank):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    torch.set_rng_state(checkpoint['torch_rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])

    training_state = {
        'n_update':         checkpoint['n_update'],
        'n_epoch':          checkpoint['n_epoch'],
        'n_batch_in_epoch': checkpoint['n_batch_in_epoch'],
        'max_len':          checkpoint['max_len'],
    }

    if rank == 0:
        print(f"Resumed from update {training_state['n_update']}  "
              f"(epoch={training_state['n_epoch']})")

    return training_state


def DDP_model(dance_model, df, h5py_paths, save_dir,
              lambda_mlm, lambda_dynamics, resume_from=None):

    model_cfg = config[MODEL_SELECT]
    train_cfg = config['training']
    total_update              = model_cfg['total_update']
    short_update              = model_cfg['short_update']
    save_per_update           = train_cfg['save_per_update']
    get_dataloader_per_update = train_cfg['get_dataloader_per_update']
    freeze_encoder_epochs     = train_cfg['freeze_encoder_epochs']
    mask_prob                 = train_cfg['mask_prob']

    dist.init_process_group("nccl")
    rank      = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    dance_model = dance_model.to(device_id)
    ddp_model   = DDP(dance_model, device_ids=[device_id], find_unused_parameters=True)

    def set_encoder_trainable(trainable):
        for param in ddp_model.module.encoder.parameters():
            param.requires_grad = trainable
        if rank == 0:
            print("Encoder UNFROZEN — full fine-tuning" if trainable
                  else "Encoder FROZEN — training heads only")

    # Encoder and MLM head use peak_lr; the randomly initialised dynamics heads use 10x.
    encoder_params    = list(ddp_model.module.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    mlm_head_unique   = [p for p in ddp_model.module.mlm_head.parameters()
                         if id(p) not in encoder_param_ids]
    dynamics_params   = (
        list(ddp_model.module.residue_dynamics_head.parameters()) +
        list(ddp_model.module.pairwise_transform.parameters()) +
        list(ddp_model.module.pairwise_middle.parameters()) +
        list(ddp_model.module.pairwise_output.parameters())
    )

    lr_group_names = ['encoder', 'mlm', 'dynamics']
    optimizer = AdamW(
        [
            {"params": encoder_params,  "lr": 1.0},
            {"params": mlm_head_unique, "lr": 1.0},
            {"params": dynamics_params, "lr": 10.0},
        ],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['epsilon'],
        weight_decay=config['optimizer']['weight_decay'],
    )
    scheduler = WarmupDecaySchedule(optimizer, MODEL_SELECT)
    scaler    = GradScaler('cuda')

    if rank == 0:
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    if resume_from == "latest":
        resume_from = os.path.join(save_dir, 'checkpoints', 'latest.pt')
    if resume_from is not None and not os.path.exists(resume_from):
        if rank == 0:
            print(f"Checkpoint not found: {resume_from}, starting from scratch")
        resume_from = None

    if resume_from is not None:
        training_state   = load_checkpoint(resume_from, ddp_model, optimizer, scheduler, scaler, rank)
        n_update         = training_state['n_update']
        n_epoch          = training_state['n_epoch']
        n_batch_in_epoch = training_state['n_batch_in_epoch']
        max_len          = training_state['max_len']
    else:
        n_update         = 0
        n_epoch          = 0
        n_batch_in_epoch = 0
        max_len          = model_cfg['max_len_short']

        if rank == 0:
            torch.save(ddp_model.module.state_dict(),
                       os.path.join(save_dir, 'checkpoints', 'update_0.pt'))

            total_params = sum(p.numel() for p in ddp_model.parameters())
            print(f"Total params: {total_params:,}")

            training_config = {
                'esm2_select': ESM2_SELECT, 'model_select': MODEL_SELECT,
                'total_update': total_update, 'short_update': short_update,
                'lambda_mlm': lambda_mlm, 'lambda_dynamics': lambda_dynamics,
                'mask_prob': mask_prob, 'total_params': total_params,
                'freeze_encoder_epochs': freeze_encoder_epochs,
            }
            with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
                json.dump(training_config, f, indent=2)

    # One "epoch" = one dataloader refresh, i.e. every `get_dataloader_per_update` updates.
    encoder_frozen = n_epoch < freeze_encoder_epochs
    set_encoder_trainable(not encoder_frozen)

    start_time = time.time()
    log        = {}

    while n_update <= total_update:

        if (n_batch_in_epoch / model_cfg[f'update_batch_{max_len}']) % get_dataloader_per_update == 0:
            log = {}
            n_epoch         += 1
            n_batch_in_epoch = 0

            if encoder_frozen and n_epoch > freeze_encoder_epochs:
                set_encoder_trainable(True)
                encoder_frozen = False

            if n_update >= short_update:
                max_len = model_cfg['max_len_long']

            three_cycle_iter = get_dataloader_cycle_iter(
                df, h5py_paths, ESM2_SELECT, max_len,
                model_cfg[f'batch_size_{max_len}'],
                n_update, device_id, mask_prob=mask_prob,
            )

            if rank == 0:
                frozen_str = "FROZEN" if encoder_frozen else "UNFROZEN"
                print(f"\n=== Epoch {n_epoch} | Update {n_update} | "
                      f"Max length: {max_len} | Encoder: {frozen_str} ===")

        mlm_correct_sum = 0
        mlm_total_sum   = 0
        update_batch    = model_cfg[f'update_batch_{max_len}']

        # One batch from each of the three source groups per step
        for source in three_cycle_iter:
            data = next(three_cycle_iter[source])

            inputs = {
                "input_ids":      data['masked_input_ids'].to(device_id, non_blocking=True),
                "attention_mask": data['attention_mask'].to(device_id, non_blocking=True),
            }
            batch = {
                'original_input_ids': data['original_input_ids'].to(device_id, non_blocking=True),
                'mask_positions':     data['mask_positions'].to(device_id, non_blocking=True),
                'attention_mask':     data['attention_mask'].to(device_id, non_blocking=True),
                'res_feat':           data['res_feat'].to(device_id, non_blocking=True),
                'pair_feat':          data['pair_feat'].to(device_id, non_blocking=True),
            }

            with autocast(dtype=torch.float16, device_type='cuda'):
                output = ddp_model(inputs)
                losses = calculate_total_loss(
                    source, output, batch,
                    lambda_mlm=lambda_mlm,
                    lambda_dynamics=lambda_dynamics,
                    num_sources=len(three_cycle_iter),
                )
                loss_batch       = losses['total_loss'] / update_batch
                mlm_correct_sum += losses['mlm_correct']
                mlm_total_sum   += losses['mlm_total']

            scaler.scale(loss_batch).backward()

            for feature in losses:
                if feature in ['mlm_correct', 'mlm_total']:
                    continue
                log[feature] = log.get(feature, 0.0) + (
                    losses[feature].item() / update_batch / train_cfg['report_per_update']
                )

        n_batch_in_epoch += 1

        if n_batch_in_epoch % update_batch == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(ddp_model.parameters(), 0.5)
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            n_update += 1

            if n_update % train_cfg['report_per_update'] == 0:
                update_time      = time.time() - start_time
                start_time       = time.time()
                avg_mlm_accuracy = (mlm_correct_sum / mlm_total_sum) if mlm_total_sum > 0 else 0.0
                lr_map = dict(zip(lr_group_names, scheduler.get_last_lr()))

                log_entry = {
                    "rank": rank, "n_epoch": n_epoch, "n_update": n_update,
                    "n_batch_in_epoch": n_batch_in_epoch, "batch_time": update_time,
                    "encoder_learning_rate": lr_map['encoder'],
                    "mlm_learning_rate":     lr_map['mlm'],
                    "dynamic_learning_rate": lr_map['dynamics'],
                    "max_len": max_len,
                    "mask_prob": mask_prob, "avg_mlm_accuracy": avg_mlm_accuracy,
                    "encoder_frozen": encoder_frozen,
                    **log,
                }

                if rank == 0:
                    print(f"Update {n_update:5d} | "
                          f"[{'FROZEN' if encoder_frozen else 'UNFROZEN'}] | "
                          f"Encoder Lr: {lr_map['encoder']:.2e} | "
                          f"Dynamic Lr: {lr_map['dynamics']:.2e} | "
                          f"MLM Acc: {avg_mlm_accuracy:.4f} | "
                          f"MLM Loss: {log.get('mlm_loss', 0):.4f} | "
                          f"Total Loss: {log.get('total_loss', 0):.4f} | "
                          f"Time: {update_time:.2f}s")

                    with open(os.path.join(save_dir, "training_log.json"), "a") as f:
                        json.dump(log_entry, f)
                        f.write('\n')

                log = {}

            if n_update % save_per_update == 0:
                save_checkpoint(
                    save_dir, ddp_model, optimizer, scheduler, scaler,
                    n_update, n_epoch, n_batch_in_epoch, max_len, rank,
                )

    save_checkpoint(
        save_dir, ddp_model, optimizer, scheduler, scaler,
        n_update, n_epoch, n_batch_in_epoch, max_len, rank, is_final=True,
    )
    if rank == 0:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Second-stage multi-task pretraining of DynamicsPLM')
    parser.add_argument('--save_dir', default=config['file_path']['save_dir'],
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--data_dir', default=config['file_path']['data_dir'],
                        help='Directory with the training CSV, loss-weight CSV and HDF5 files')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint, or "latest" to auto-resume')
    parser.add_argument('--lambda_mlm', type=float, default=config['training']['lambda_mlm'],
                        help='Weight of the MLM loss in Eq. (1) (default: 0.7)')
    parser.add_argument('--lambda_dynamics', type=float, default=config['training']['lambda_dynamics'],
                        help='Weight of the dynamics loss in Eq. (1) (default: 0.3)')
    args = parser.parse_args()

    load_loss_weight(args.data_dir)
    h5py_paths = get_h5py_paths(args.data_dir)

    df = pd.read_csv(os.path.join(args.data_dir, config['file_path']['train_df']))
    df = df[df['label'] == 'train']

    dance_model = DynamicsPLM(ESM2_SELECT)
    dance_model.train()

    print(f"Train proteins : {len(df):,}")
    print(f"Save dir       : {args.save_dir}")
    print(f"Resume         : {args.resume or 'no'}")
    print(f"lambda_mlm / lambda_dynamics : {args.lambda_mlm} / {args.lambda_dynamics}")

    DDP_model(
        dance_model, df, h5py_paths, args.save_dir,
        lambda_mlm=args.lambda_mlm,
        lambda_dynamics=args.lambda_dynamics,
        resume_from=args.resume,
    )
