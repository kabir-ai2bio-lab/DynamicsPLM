import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    "file_path": {
        "save_dir": os.path.join(PROJECT_ROOT, "save_dir"),
        # Local copy of https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties
        "data_dir": os.path.join(PROJECT_ROOT, "dataset", "protein_dynamic_properties"),
        "train_df": "training_test_data_sequence.csv",
        "loss_weight": "feature_weight.csv",
        "h5_files": [
            "feature_mdCATH_ATLAS_GPCRmd_PED_IDRome.h5",
            "feature_proteinflow_1.h5",
            "feature_proteinflow_2.h5",
        ],
    },

    "training": {
        "random_seed": 42,
        "dropout": 0.1,
        "save_per_update": 1000,
        "report_per_update": 20,
        "get_dataloader_per_update": 200,
        "freeze_encoder_epochs": 2,
        "mask_prob": 0.15,

        # Eq. (1): L_total = lambda_mlm * L_MLM + lambda_dynamics * L_dynamics
        # Published DynamicsPLM uses 0.7 / 0.3 (paper Section V-C).
        "lambda_mlm": 0.7,
        "lambda_dynamics": 0.3,

        "res_feature_idx": {
            "sasa_mean": 0,
            "sasa_std": 1,
            "rmsf_nor": 2,
            "ss": list(range(3, 11)),
            "chi": list(range(11, 23)),
            "phi": list(range(23, 35)),
            "psi": list(range(35, 47)),
            "nma_res1": 47,
            "nma_res2": 48,
            "nma_res3": 49,
        },
        "pair_feature_idx": {
            "vdw": 0,
            "hbbb": 1,
            "hbsb": 2,
            "hbss": 3,
            "hp": 4,
            "sb": 5,
            "pc": 6,
            "ps": 7,
            "ts": 8,
            "corr": 9,
            "nma_pair1": 10,
            "nma_pair2": 11,
            "nma_pair3": 12,
        },
    },

    "optimizer": {
        "peak_lr": 1e-5,
        "epsilon": 1e-8,
        "betas": (0.9, 0.98),
        "weight_decay": 0.01,
        "warmup_step": 700,
        "decay_step_percent": 0.9,
    },

    "dynamics_plm": {
        "max_len_short": 256,
        "max_len_long": 1024,

        "total_update": 20_000,
        "short_update": 15_000,

        "batch_size_256": 16,
        "update_batch_256": 2,

        "batch_size_1024": 2,
        "update_batch_1024": 16,
    },

    "model_35M": {
        "model_id": "facebook/esm2_t12_35M_UR50D",

        "atten_dim": 240,
        "embed_dim": 480,

        "pair_out_dim": 13,
        "res_out_dim": 50,
    },
}
