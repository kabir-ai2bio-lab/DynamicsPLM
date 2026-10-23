"""
Zero-shot mutation scoring shared by the ProteinGym and Tsuboyama benchmarks
(paper Section IV-C).

  MLM score       (Eq. 3) : masked-marginal log p(mt) - log p(wt), summed over
                            mutated sites. L forward passes per wild type,
                            then an O(1) lookup per variant.
  Dynamics score  (Eq. 4) : per-feature relative change |y_mt - y_wt| / (|y_wt| + eps),
                            averaged over residues / residue pairs, quantile-
                            normalised across variants, then geometric mean.
                            Higher = more disruptive.
  Combined score  (Eq. 5) : z(MLM) - z(Dyn) when N >= 30 variants, else MLM - Dyn.

Model types:
  dynamicsplm : MLM + dynamics + combined
  esm2        : MLM only (pretrained ESM2 baseline)
  esmdance    : dynamics only (ESMDance baseline)
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, zscore
from tqdm import tqdm

import util

EPSILON         = 1e-2   # relative-change denominator guard (Eq. 4)
MIN_N_FOR_ZSCORE = 30    # Eq. 5
MAX_SEQ_LEN     = 1022   # pairwise head is O(L^2); no windowed inference
PAIRWISE_KEYS   = set(util.config['training']['pair_feature_idx'].keys())

USES_MLM      = {'dynamicsplm', 'esm2'}
USES_DYNAMICS = {'dynamicsplm', 'esmdance'}


# ─────────────────────────────────────────────────────────────────────────────
# MLM (Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def _get_optimal_window(pos, total_len, window=1024):
    """Return (start, end) token-index slice of length `window` centered on `pos`."""
    half  = window // 2
    start = max(0, pos - half)
    end   = min(total_len, start + window)
    start = max(0, end - window)
    return start, end


@torch.no_grad()
def precompute_masked_mlm_log_probs(model, tokenizer, sequence, device,
                                    model_type='dynamicsplm', max_len=1024):
    """
    Mask each position in turn and store the log-probabilities at that position.

    Returns a (total_len, vocab_size) CPU tensor indexed by token position
    (CLS at 0, EOS at total_len-1).
    """
    model.eval()
    wt_ids    = tokenizer(sequence, return_tensors="pt",
                          add_special_tokens=True)['input_ids']
    total_len = wt_ids.size(1)
    vocab_size = model.config.vocab_size if model_type == 'esm2' else model.vocab_size

    log_prob_matrix = torch.zeros(total_len, vocab_size)

    for i in tqdm(range(1, total_len - 1), desc="  Pre-computing MLM", leave=False):
        masked = wt_ids.clone()
        masked[0, i] = tokenizer.mask_token_id

        if total_len > max_len:
            start, end = _get_optimal_window(i, total_len, max_len)
            window_ids = masked[:, start:end]
            local_i    = i - start
        else:
            window_ids = masked
            local_i    = i

        window_ids     = window_ids.to(device)
        attention_mask = torch.ones_like(window_ids)

        if model_type == 'esm2':
            logits = model(input_ids=window_ids, attention_mask=attention_mask).logits
        else:
            inputs = {'input_ids': window_ids, 'attention_mask': attention_mask}
            logits = model(inputs, return_mlm=True,
                           return_res_dynamics=False, return_pair_dynamics=False)['mlm_logits']

        log_prob_matrix[i] = F.log_softmax(logits[0], dim=-1)[local_i].cpu()

    return log_prob_matrix


def score_mlm_variant(log_prob_matrix, tokenizer, wildtype_seq, mutant_seq):
    """Handles single- and multi-site substitutions."""
    score = 0.0
    for pos, (wt, mt) in enumerate(zip(wildtype_seq, mutant_seq)):
        if wt == mt:
            continue
        token_pos = pos + 1   # +1 for CLS
        wt_id = tokenizer.convert_tokens_to_ids(wt)
        mt_id = tokenizer.convert_tokens_to_ids(mt)
        score += log_prob_matrix[token_pos, mt_id].item()
        score -= log_prob_matrix[token_pos, wt_id].item()
    return score


def mutation_string(wt_seq, mt_seq):
    muts = [f"{wt}{i+1}{mt}" for i, (wt, mt) in enumerate(zip(wt_seq, mt_seq)) if wt != mt]
    return ','.join(muts) if muts else 'wt'


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics disruption (Eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_dynamics_predictions(model, tokenizer, seq, device, model_type='dynamicsplm'):
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if model_type == 'esmdance':
        output = model(inputs)   # residue + pairwise features by default
    else:
        output = model(inputs, return_mlm=False,
                       return_res_dynamics=True, return_pair_dynamics=True)
    return {k: v.cpu().float() for k, v in output.items()}


def compute_feature_relative_change(wt_preds, mt_preds):
    """Mean relative change per feature over all residues / residue pairs
    (CLS and EOS stripped)."""
    scores = {}
    for key in wt_preds:
        wt  = wt_preds[key]
        mt  = mt_preds[key]
        rel = (mt - wt).abs() / (wt.abs() + EPSILON)
        rel = rel.squeeze(0)        # (1, L+2, ...) → (L+2, ...)
        if key in PAIRWISE_KEYS:
            rel = rel[1:-1, 1:-1]   # (L+2, L+2) → (L, L)
        else:
            rel = rel[1:-1]         # (L+2,) or (L+2, n_feat) → (L,) or (L, n_feat)
        scores[key] = rel.mean().item()
    return scores


def quantile_normalization(features):
    features        = np.array(features, dtype=np.float64)
    ranks           = np.argsort(np.argsort(features, axis=0), axis=0)
    sorted_features = np.sort(features, axis=0)
    rank_means      = np.mean(sorted_features, axis=1)
    normalised      = np.zeros_like(features)
    for j in range(features.shape[1]):
        normalised[:, j] = rank_means[ranks[:, j]]
    return normalised


def geometric_mean_rows(matrix, eps=1e-8):
    return np.exp(np.mean(np.log(matrix + eps), axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# Combined score (Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def combine_scores(mlm_score, dynamics_score):
    mlm_score      = np.asarray(mlm_score, dtype=np.float64)
    dynamics_score = np.asarray(dynamics_score, dtype=np.float64)
    if len(mlm_score) >= MIN_N_FOR_ZSCORE:
        return zscore(mlm_score) - zscore(dynamics_score)
    return mlm_score - dynamics_score


# ─────────────────────────────────────────────────────────────────────────────
# Per-protein scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_protein(model, tokenizer, wt_seq, dms_df, device, model_type,
                  protein_name="protein"):
    """
    Score every substitution variant of one protein.

    dms_df needs columns `mutated_sequence` and `DMS_score`.
    Returns a DataFrame with mutation, mutated_sequence, dms_score and the
    available mlm_score / dynamics_score / combined_score, or None.
    """
    model.eval()
    do_mlm = model_type in USES_MLM
    do_dyn = model_type in USES_DYNAMICS

    mlm_log_probs = None
    wt_preds      = None
    if do_mlm:
        mlm_log_probs = precompute_masked_mlm_log_probs(
            model, tokenizer, wt_seq, device, model_type=model_type)
    if do_dyn:
        wt_preds = get_dynamics_predictions(model, tokenizer, wt_seq, device, model_type)
        feature_names = list(wt_preds.keys())

    rows = []
    for _, row in tqdm(dms_df.iterrows(), total=len(dms_df),
                       desc=f"  {protein_name}", leave=False):
        mt_seq = row["mutated_sequence"]
        if len(mt_seq) != len(wt_seq):   # substitutions only
            continue

        entry = {
            "mutation":         mutation_string(wt_seq, mt_seq),
            "mutated_sequence": mt_seq,
            "dms_score":        row["DMS_score"],
        }
        if do_mlm:
            entry["mlm_score"] = score_mlm_variant(mlm_log_probs, tokenizer, wt_seq, mt_seq)
        if do_dyn:
            mt_preds    = get_dynamics_predictions(model, tokenizer, mt_seq, device, model_type)
            feat_change = compute_feature_relative_change(wt_preds, mt_preds)
            entry.update({f"dyn_{k}": feat_change[k] for k in feature_names})
        rows.append(entry)

    if not rows:
        return None

    result_df = pd.DataFrame(rows)

    if do_dyn:
        dyn_cols = [f"dyn_{k}" for k in feature_names]
        result_df["dynamics_score"] = geometric_mean_rows(
            quantile_normalization(result_df[dyn_cols].values))
        result_df.drop(columns=dyn_cols, inplace=True)

    if do_mlm and do_dyn:
        result_df["combined_score"] = combine_scores(
            result_df["mlm_score"], result_df["dynamics_score"])

    return result_df


def spearman_summary(result_df):
    """Spearman rho of each available score against the experimental label.
    rho_dynamics is negative when disruption tracks lower fitness/stability."""
    y   = result_df["dms_score"].values
    out = {}
    for col, key in [("mlm_score", "rho_mlm"),
                     ("dynamics_score", "rho_dynamics"),
                     ("combined_score", "rho_combined")]:
        if col in result_df.columns:
            out[key] = spearmanr(result_df[col], y).statistic
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop (resumable)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(model, tokenizer, proteins, device, model_type, out_dir):
    """
    proteins : iterable of dicts {'name', 'wt_seq', 'dms_df'} where dms_df is a
               DataFrame or a path to a CSV (read lazily).

    Writes {out_dir}/summary.csv (one row per protein, appended as it goes so a
    restarted job resumes) and {out_dir}/per_protein/{name}.csv.
    """
    summary_path    = os.path.join(out_dir, "summary.csv")
    per_protein_dir = os.path.join(out_dir, "per_protein")
    os.makedirs(per_protein_dir, exist_ok=True)

    processed = set()
    if os.path.exists(summary_path):
        processed = set(pd.read_csv(summary_path)["protein"])
        print(f"Resuming: {len(processed)} proteins already in {summary_path}")
    first_write = not os.path.exists(summary_path)

    for prot in proteins:
        name, wt_seq, dms_df = prot['name'], prot['wt_seq'], prot['dms_df']

        if name in processed:
            continue
        if len(wt_seq) > MAX_SEQ_LEN:
            print(f"Skipping {name}: length {len(wt_seq)} > {MAX_SEQ_LEN}")
            continue
        if isinstance(dms_df, str):
            if not os.path.exists(dms_df):
                print(f"Skipping {name}: file not found {dms_df}")
                continue
            dms_df = pd.read_csv(dms_df)

        print(f"\n[{name}]  variants: {len(dms_df)}")
        result_df = score_protein(model, tokenizer, wt_seq, dms_df, device,
                                  model_type, protein_name=name)
        if result_df is None:
            print("  Skipping: no valid variants")
            continue

        rhos = spearman_summary(result_df)
        print("  " + "  ".join(f"{k}={v:+.4f}" for k, v in rhos.items()))

        pd.DataFrame([{"protein": name, "n_variants": len(result_df), **rhos}]).to_csv(
            summary_path, mode='a', index=False, header=first_write)
        first_write = False

        safe_name = name.replace('|', '_').replace('/', '_')
        result_df.to_csv(os.path.join(per_protein_dir, f"{safe_name}.csv"), index=False)

    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        print(f"\n{'='*60}\n  SUMMARY  |  {model_type}  |  {len(summary)} proteins")
        for col in ["rho_mlm", "rho_dynamics", "rho_combined"]:
            if col in summary.columns:
                print(f"  {col:<14} mean = {summary[col].mean():+.4f}   "
                      f"median = {summary[col].median():+.4f}")
        print(f"  Results → {summary_path}\n{'='*60}")
