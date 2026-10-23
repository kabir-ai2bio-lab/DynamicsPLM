"""
Shared utilities for the DynamicsPLM evaluation scripts:
path constants, model loading and output-directory management.
"""
import importlib.util as _importlib_util
import os
import sys
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

# ── Project paths ─────────────────────────────────────────────────────────────
_EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))   # …/evaluation/
PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
DATASET_DIR  = os.path.join(PROJECT_ROOT, 'dataset')
SAVE_DIR     = os.path.join(PROJECT_ROOT, 'save_dir')

DMS_DIR        = os.path.join(DATASET_DIR, 'DMS_ProteinGym_substitutions')
DMS_REFERENCE  = os.path.join(DMS_DIR, 'DMS_substitutions.csv')
TSUBOYAMA_CSV  = os.path.join(DATASET_DIR, 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
DEFAULT_CHECKPOINT = os.path.join(SAVE_DIR, 'checkpoints', 'best.pt')

# Make model/ importable from evaluation/
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# ESMDance baseline code: clone https://github.com/ShenLab/SeqDance into evaluation/SeqDance
SEQDANCE_DIR = os.path.join(_EVAL_DIR, 'SeqDance', 'model')

from config import config          # noqa: E402  (must follow sys.path setup)
from model  import DynamicsPLM     # noqa: E402


# ── Architecture fingerprints ─────────────────────────────────────────────────

_DPLM_SIGNATURE  = {"residue_dynamics_head.0.weight", "pairwise_output.weight"}
_DANCE_SIGNATURE = {"res_pred_nn.0.weight", "pair_pred_linear.weight"}


def _assert_architecture(state, required, forbidden, label):
    ckpt_keys = set(state.keys())
    missing   = required - ckpt_keys
    intruding = forbidden & ckpt_keys
    if missing or intruding:
        parts = []
        if missing:
            parts.append(f"expected keys not found: {sorted(missing)}")
        if intruding:
            parts.append(f"keys from wrong architecture present: {sorted(intruding)}")
        raise ValueError(
            f"Architecture mismatch in {label} checkpoint — "
            + "; ".join(parts)
            + ". Verify you are passing the correct checkpoint for each model."
        )


def _load_state_dict(checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(checkpoint_path, device="cpu")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return raw.get("model_state_dict", raw)


def _report_key_mismatch(missing, unexpected):
    if missing:
        print(f"    [WARN] Missing keys    ({len(missing)}): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"    [WARN] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")


# ── Device / tokenizer ────────────────────────────────────────────────────────

def get_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_tokenizer(esm2_select='model_35M'):
    return AutoTokenizer.from_pretrained(config[esm2_select]['model_id'])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_dynamics_plm(checkpoint_path, esm2_select, device):
    print(f"  Loading DynamicsPLM: {checkpoint_path}")
    model = DynamicsPLM(esm2_select)
    state = _load_state_dict(checkpoint_path)
    _assert_architecture(state, _DPLM_SIGNATURE, _DANCE_SIGNATURE, "DynamicsPLM")
    _report_key_mismatch(*model.load_state_dict(state, strict=False))
    return model.to(device).eval()


def load_esm2_mlm(esm2_select, device):
    model_id = config[esm2_select]['model_id']
    print(f"  Loading ESM2 MLM ({model_id})")
    return EsmForMaskedLM.from_pretrained(model_id).to(device).eval()


def _import_esm_wrap():
    """Load ESMwrap from SeqDance/model/model.py by absolute path using importlib,
    so it never collides with DynamicsPLM's model/model.py on sys.path."""
    config_path = os.path.join(SEQDANCE_DIR, "config.py")
    model_path  = os.path.join(SEQDANCE_DIR, "model.py")

    for p in (config_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"SeqDance source file not found: {p}\n"
                f"Clone https://github.com/ShenLab/SeqDance into evaluation/SeqDance"
            )

    config_spec = _importlib_util.spec_from_file_location("_seqdance_config", config_path)
    config_mod  = _importlib_util.module_from_spec(config_spec)
    sys.modules["_seqdance_config"] = config_mod
    config_spec.loader.exec_module(config_mod)

    # Temporarily shadow 'config' in sys.modules so SeqDance's model.py resolves
    # `from config import config` to SeqDance's config, not DynamicsPLM's.
    prev_config = sys.modules.get("config")
    sys.modules["config"] = config_mod
    try:
        model_spec = _importlib_util.spec_from_file_location("_seqdance_model", model_path)
        model_mod  = _importlib_util.module_from_spec(model_spec)
        sys.modules["_seqdance_model"] = model_mod
        model_spec.loader.exec_module(model_mod)
    finally:
        if prev_config is not None:
            sys.modules["config"] = prev_config
        else:
            sys.modules.pop("config", None)

    return model_mod.ESMwrap


def load_esmdance(checkpoint_path, esm2_select, device):
    """Load ESMDance from a local .safetensors/.pt file, or from HuggingFace
    (ChaoHou/ESMDance) when checkpoint_path is None."""
    ESMwrap = _import_esm_wrap()
    if checkpoint_path is None:
        print("  Loading ESMDance from HuggingFace (ChaoHou/ESMDance)")
        model = ESMwrap(esm2_select, 'esmdance').from_pretrained("ChaoHou/ESMDance")
        return model.to(device).eval()

    print(f"  Loading ESMDance: {checkpoint_path}")
    model = ESMwrap(esm2_select, 'esmdance')
    state = _load_state_dict(checkpoint_path)
    _assert_architecture(state, _DANCE_SIGNATURE, _DPLM_SIGNATURE, 'ESMDance')
    _report_key_mismatch(*model.load_state_dict(state, strict=False))
    return model.to(device).eval()


def load_model(model_name, checkpoint, esm2_select, device):
    if model_name == 'dynamicsplm':
        return load_dynamics_plm(checkpoint or DEFAULT_CHECKPOINT, esm2_select, device)
    if model_name == 'esm2':
        return load_esm2_mlm(esm2_select, device)
    if model_name == 'esmdance':
        return load_esmdance(checkpoint, esm2_select, device)
    raise ValueError(f"Unknown model: {model_name}")


# ── Output directory management ───────────────────────────────────────────────

def make_output_dir(save_dir, *parts):
    """Create {save_dir}/evaluation/{parts...}/ and return the path."""
    path = os.path.join(save_dir, 'evaluation', *parts)
    os.makedirs(path, exist_ok=True)
    return path


def add_common_args(parser):
    parser.add_argument('--model', choices=['dynamicsplm', 'esm2', 'esmdance'],
                        default='dynamicsplm',
                        help='dynamicsplm: MLM + dynamics combined score (Eq. 5); '
                             'esm2: MLM-only baseline; esmdance: dynamics-only baseline')
    parser.add_argument('--checkpoint', default=None,
                        help=f'Model weights. dynamicsplm default: {DEFAULT_CHECKPOINT}; '
                             'esmdance default: HuggingFace ChaoHou/ESMDance; ignored for esm2')
    parser.add_argument('--esm2_select', default='model_35M',
                        help='ESM2 backbone config key (default: model_35M)')
    parser.add_argument('--save_dir', default=SAVE_DIR,
                        help='Outputs go to {save_dir}/evaluation/...')
    parser.add_argument('--device', default=None,
                        help='cuda / cpu / mps (default: auto-detect)')
    return parser
