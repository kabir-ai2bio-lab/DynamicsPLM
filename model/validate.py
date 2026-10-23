"""
Score saved checkpoints on the held-out evaluation proteins and select the one
with the lowest mean dynamics loss (SI S1.2). The selected weights are written
to {save_dir}/checkpoints/best.pt, which the evaluation scripts load by default.
"""
import os
import pandas as pd
import json
import torch

from config import config
from model import DynamicsPLM
from dataset import ProteinDatasetMLM
import train
from train import calculate_dynamics_loss, calculate_mlm_loss

SOURCE_GROUPS = {
    "ATLAS_GPCRmd_PED_mdCATH": ["ATLAS", "GPCRmd", "PED", "mdCATH"],
    "IDRome":                  ["IDRome"],
    "Proteinflow":             ["proteinflow_pdb", "proteinflow_sabdab"],
}


def get_mean_losses(dance_model, three_dataset, device):
    mean_loss = {}
    mlm_correct_total = 0
    mlm_token_total   = 0

    for source, dataset in three_dataset.items():
        n_samples = len(dataset)
        if n_samples == 0:
            print(f"  [WARN] Source '{source}' has 0 samples, skipping.")
            continue

        source_mlm_loss = 0.0

        for data in dataset:
            data = {k: v.unsqueeze(0) for k, v in data.items()}

            inputs = {
                "input_ids":      data["masked_input_ids"].to(device, non_blocking=True),
                "attention_mask": data["attention_mask"].to(device, non_blocking=True),
            }
            res_feat  = data["res_feat"].to(device, non_blocking=True)
            pair_feat = data["pair_feat"].to(device, non_blocking=True)

            with torch.no_grad():
                output = dance_model(inputs)

                mlm_loss, correct, total = calculate_mlm_loss(
                    output['mlm_logits'],
                    data['original_input_ids'].to(device, non_blocking=True),
                    data['mask_positions'].to(device, non_blocking=True),
                    data['attention_mask'].to(device, non_blocking=True),
                )
                source_mlm_loss   += mlm_loss.item()
                mlm_correct_total += correct
                mlm_token_total   += total

                dyn_losses = calculate_dynamics_loss(source, output, res_feat, pair_feat)

            for feature, value in dyn_losses.items():
                mean_loss[feature] = mean_loss.get(feature, 0.0) + value.item() / n_samples

        mean_loss[f'{source}_mlm_loss'] = source_mlm_loss / n_samples

    if mlm_token_total > 0:
        mean_loss['mlm_accuracy'] = mlm_correct_total / mlm_token_total

    return mean_loss


def load_model_weights(checkpoint_path, dance_model, device):
    # weights_only=False: full checkpoints contain numpy/Python rng states
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw

    missing, unexpected = dance_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    dance_model = dance_model.to(device)
    dance_model.eval()
    return dance_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DynamicsPLM checkpoints on held-out dynamics loss")
    parser.add_argument("--save_dir",   default=config["file_path"]["save_dir"],
                        help="Training output directory (contains checkpoints/)")
    parser.add_argument("--data_dir",   default=config["file_path"]["data_dir"],
                        help="Directory with the training CSV, loss-weight CSV and HDF5 files")
    parser.add_argument("--max_len",    type=int, default=1024,   help="Max sequence length")
    parser.add_argument("--chk_start",  type=int, default=1_000,  help="First checkpoint update")
    parser.add_argument("--chk_end",    type=int, default=20_000, help="Last checkpoint update")
    parser.add_argument("--chk_step",   type=int, default=1_000,  help="Step between checkpoints")
    parser.add_argument("--single_chk", default=None,             help="Evaluate a single checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train.load_loss_weight(args.data_dir)
    h5py_paths = train.get_h5py_paths(args.data_dir)

    df = pd.read_csv(os.path.join(args.data_dir, config["file_path"]["train_df"]))
    df = df[df["label"] == "test"].reset_index(drop=True)
    print(f"Held-out proteins: {len(df)}")

    # N-terminal window for reproducibility (SI S1.1)
    three_dataset = {
        name: ProteinDatasetMLM(
            df[df["source"].isin(sources)].reset_index(drop=True),
            h5py_paths, train.ESM2_SELECT, args.max_len, start_from_N_terminal=True,
        )
        for name, sources in SOURCE_GROUPS.items()
    }
    for src, ds in three_dataset.items():
        print(f"  {src}: {len(ds)} samples")

    dance_model = DynamicsPLM(train.ESM2_SELECT)

    out_path = os.path.join(args.save_dir, f"validation_maxlen_{args.max_len}.json")
    print(f"\nResults will be appended to: {out_path}\n")

    if args.single_chk:
        checkpoints = [("custom", args.single_chk)]
    else:
        checkpoints = [
            (chk, os.path.join(args.save_dir, "checkpoints", f"update_{chk}.pt"))
            for chk in range(args.chk_start, args.chk_end + 1, args.chk_step)
        ]

    best_chk_path = None
    best_dyn_loss = float("inf")

    for chk_id, chk_path in checkpoints:
        if not os.path.exists(chk_path):
            print(f"[SKIP] Checkpoint not found: {chk_path}")
            continue

        print(f"Evaluating checkpoint {chk_id}: {os.path.basename(chk_path)}")

        dance_model = load_model_weights(chk_path, dance_model, device)
        mean_loss   = get_mean_losses(dance_model, three_dataset, device)
        mean_loss["checkpoint"] = chk_id

        source_dyn_losses = [mean_loss[f"{src}_dynamics_loss"] for src in three_dataset
                             if f"{src}_dynamics_loss" in mean_loss]
        source_mlm_losses = [mean_loss[f"{src}_mlm_loss"] for src in three_dataset
                             if f"{src}_mlm_loss" in mean_loss]
        mean_loss["overall_dynamics_loss"] = sum(source_dyn_losses) / len(source_dyn_losses)
        mean_loss["overall_mlm_loss"]      = sum(source_mlm_losses) / len(source_mlm_losses)

        print(f"  {'Overall':30s} dynamics loss: {mean_loss['overall_dynamics_loss']:.4f}")
        print(f"  {'Overall':30s} mlm loss:      {mean_loss['overall_mlm_loss']:.4f}")
        print(f"  {'Overall':30s} mlm accuracy:  {mean_loss.get('mlm_accuracy', 0.0):.4f}")

        if mean_loss["overall_dynamics_loss"] < best_dyn_loss:
            best_dyn_loss = mean_loss["overall_dynamics_loss"]
            best_chk_path = chk_path

        with open(out_path, "a") as f:
            json.dump(mean_loss, f)
            f.write("\n")

    if best_chk_path is not None:
        best_out = os.path.join(args.save_dir, "checkpoints", "best.pt")
        raw = torch.load(best_chk_path, map_location="cpu", weights_only=False)
        torch.save(raw.get("model_state_dict", raw), best_out)
        print(f"\nBest checkpoint: {os.path.basename(best_chk_path)}  |  "
              f"dynamics loss: {best_dyn_loss:.4f}")
        print(f"Weights saved to: {best_out}")
    print(f"Results saved to: {out_path}")
