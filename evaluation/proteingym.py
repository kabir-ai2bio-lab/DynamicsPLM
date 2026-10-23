"""
Zero-shot mutation effect prediction on the ProteinGym DMS substitution
benchmark (paper Section V-A / V-B1, Fig. 2-3, Table S3).

Proteins longer than 1022 residues and indel variants are skipped, giving the
201 assays reported in the paper.

Outputs:
  {save_dir}/evaluation/proteingym/{model}/summary.csv
  {save_dir}/evaluation/proteingym/{model}/per_protein/{DMS_id}.csv

Examples (run from the project root):
  python evaluation/proteingym.py --model dynamicsplm
  python evaluation/proteingym.py --model esm2
  python evaluation/proteingym.py --model esmdance
  python evaluation/proteingym.py --model dynamicsplm --protein_id A4GRB6_PSEAI_Chen_2020
"""
import argparse
import os

import pandas as pd

import util
import scoring


def iter_proteingym(reference_csv, dms_dir, protein_id=None):
    reference_df = pd.read_csv(reference_csv)
    if protein_id is not None:
        reference_df = reference_df[reference_df["DMS_id"] == protein_id]
        if reference_df.empty:
            raise ValueError(f"'{protein_id}' not found in {reference_csv}")
    for _, ref in reference_df.iterrows():
        yield {
            'name':   ref['DMS_id'],
            'wt_seq': ref['target_seq'],
            'dms_df': os.path.join(dms_dir, ref['DMS_filename']),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot scoring on ProteinGym DMS substitutions")
    util.add_common_args(parser)
    parser.add_argument("--reference_csv", default=util.DMS_REFERENCE,
                        help="ProteinGym reference CSV (DMS_substitutions.csv)")
    parser.add_argument("--dms_dir", default=util.DMS_DIR,
                        help="Directory with the per-assay ProteinGym CSVs")
    parser.add_argument("--protein_id", default=None,
                        help="Score a single DMS_id instead of the full benchmark")
    args = parser.parse_args()

    device    = util.get_device(args.device)
    tokenizer = util.get_tokenizer(args.esm2_select)
    model     = util.load_model(args.model, args.checkpoint, args.esm2_select, device)
    out_dir   = util.make_output_dir(args.save_dir, 'proteingym', args.model)

    print(f"Device : {device}\nModel  : {args.model}\nOutput : {out_dir}")

    scoring.run_benchmark(
        model, tokenizer,
        iter_proteingym(args.reference_csv, args.dms_dir, args.protein_id),
        device, args.model, out_dir,
    )
