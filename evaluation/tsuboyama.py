"""
Zero-shot folding-stability prediction on the Tsuboyama 2023 mega-scale
dataset (paper Section V-B2 / V-C, Table I, Table S4, Fig. 4, Fig. S2).

  --label ddG : ddG_ML (412 proteins; Table S4, Fig. 4, Table I ddG column)
  --label dG  : dG_ML  (Table I dG column)

Filtering: substitution variants only (indels dropped), rows with a missing
label dropped, label averaged across codon replicates of the same sequence,
proteins with < 2 variants or > 1022 residues skipped.

Outputs:
  {save_dir}/evaluation/tsuboyama_{label}/{model}/summary.csv
  {save_dir}/evaluation/tsuboyama_{label}/{model}/per_protein/{WT_name}.csv

Examples (run from the project root):
  python evaluation/tsuboyama.py --model dynamicsplm --label ddG
  python evaluation/tsuboyama.py --model dynamicsplm --label dG
  python evaluation/tsuboyama.py --model esmdance    --label ddG
  python evaluation/tsuboyama.py --model dynamicsplm --label ddG --protein 1YU5.pdb
"""
import argparse

import pandas as pd

import util
import scoring

# Proteins that have no mut_type == 'wt' row in Tsuboyama; WT sequence provided manually.
_WT_OVERRIDES = {
    '1A0N.pdb': 'VTLFVALYDYEARTEDDLSFHKGEKFQILNSSEGDWWEARSLTTGETGYIPSNYVAPV',
}


def load_tsuboyama_proteins(tsuboyama_csv, label='ddG', protein=None):
    """Parse the Tsuboyama CSV into a list of {'name', 'wt_seq', 'dms_df'} dicts."""
    col = f'{label}_ML'
    df  = pd.read_csv(tsuboyama_csv, low_memory=False)

    df = df[
        (df[col] != '-') &
        (~df['mut_type'].str.contains('ins|del', na=False))
    ].copy()
    # dG_ML saturates at the assay limits; clamp the sentinel strings
    df[col] = df[col].replace({'<-1': '-1.0', '>5': '5.0'}).astype(float)

    proteins = []
    for wt_name, group in df.groupby('WT_name'):
        if protein is not None and wt_name != protein:
            continue

        wt_rows = group[group['mut_type'] == 'wt']
        if not wt_rows.empty:
            wt_seq = wt_rows.iloc[0]['aa_seq']
        elif label == 'dG' and wt_name in _WT_OVERRIDES:
            wt_seq = _WT_OVERRIDES[wt_name]
        else:
            continue

        variants = group[
            (group['mut_type'] != 'wt') &
            (group['aa_seq'].str.len() == len(wt_seq))
        ]
        variants = (variants.groupby('aa_seq', as_index=False)[col]
                             .mean()
                             .rename(columns={'aa_seq': 'mutated_sequence', col: 'DMS_score'}))
        if len(variants) < 2:
            continue

        proteins.append({'name': wt_name, 'wt_seq': wt_seq, 'dms_df': variants})

    if protein is not None and not proteins:
        raise ValueError(f"Protein '{protein}' not found in {tsuboyama_csv}")
    return proteins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot scoring on the Tsuboyama 2023 stability dataset")
    util.add_common_args(parser)
    parser.add_argument("--tsuboyama_csv", default=util.TSUBOYAMA_CSV,
                        help="Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    parser.add_argument("--label", choices=['ddG', 'dG'], default='ddG',
                        help="Experimental label to correlate with (default: ddG)")
    parser.add_argument("--protein", default=None,
                        help="Score a single WT_name (e.g. 1YU5.pdb) instead of the full set")
    args = parser.parse_args()

    device    = util.get_device(args.device)
    tokenizer = util.get_tokenizer(args.esm2_select)
    model     = util.load_model(args.model, args.checkpoint, args.esm2_select, device)
    out_dir   = util.make_output_dir(args.save_dir, f'tsuboyama_{args.label}', args.model)

    proteins = load_tsuboyama_proteins(args.tsuboyama_csv, args.label, args.protein)
    print(f"Device   : {device}\nModel    : {args.model}\nLabel    : {args.label}_ML")
    print(f"Proteins : {len(proteins)}\nOutput   : {out_dir}")

    scoring.run_benchmark(model, tokenizer, proteins, device, args.model, out_dir)
