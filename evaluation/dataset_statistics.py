"""
Exploratory data analysis for the DynamicsPLM training dataset.

Two completely separate HDF5 feature schemas exist:

  Group A — ATLAS / GPCRmd / PED / mdCATH / IDRome
    res_feature  : (L, 47)  — SASA×2, RMSF, SS×8, chi×12, phi×12, psi×12
    pair_feature : (L, L, 10) — vdw, hbbb, hbsb, hbss, hp, sb, pc, ps, ts, corr

  Group B — Proteinflow (pdb + sabdab)
    res_feature  : (L, 3)   — NMA_res1, NMA_res2, NMA_res3
    pair_feature : (L, L, 3) — NMA_pair1, NMA_pair2, NMA_pair3

Sentinel value for missing data is -1 in both groups.
Valid-row mask uses channel-0 != -1 (matching dataset.py / train.py).

Reproduces the dataset counts in paper Section III-A and Table S1.

Run from project root:
    python evaluation/dataset_statistics.py [--data_dir DIR] [--figures_dir save_dir/figures] [--split all|train|test]
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

# ── constants ─────────────────────────────────────────────────────────────────

from config import config  # noqa: E402

DATA_DIR = Path(config["file_path"]["data_dir"])

# Sources that use the (L,47) / (L,L,10) schema
MD_IDR_SOURCES = {"ATLAS", "GPCRmd", "PED", "mdCATH", "IDRome"}
# Sources that use the (L,3) / (L,L,3) NMA-only schema
PROTEINFLOW_SOURCES = {"proteinflow_pdb", "proteinflow_sabdab"}

# Group A — residue feature column layout (47 cols)
RES_A = {
    "sasa_mean": [0],
    "sasa_std":  [1],
    "rmsf":      [2],
    "ss":        list(range(3, 11)),    # 8-class one-hot
    "chi":       list(range(11, 23)),   # sin/cos encoding
    "phi":       list(range(23, 35)),
    "psi":       list(range(35, 47)),
}

# Group A — pairwise feature channel layout (10 channels)
PAIR_A = {
    "vdw":  0, "hbbb": 1, "hbsb": 2, "hbss": 3,
    "hp":   4, "sb":   5, "pc":   6, "ps":   7,
    "ts":   8, "corr": 9,
}

# Group B — residue (3 cols = NMA modes)
RES_B = {"nma_res1": 0, "nma_res2": 1, "nma_res3": 2}

# Group B — pairwise (3 channels = NMA pairwise)
PAIR_B = {"nma_pair1": 0, "nma_pair2": 1, "nma_pair3": 2}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


# ── incremental statistics ────────────────────────────────────────────────────

@dataclass
class WelfordStats:
    """Online mean/variance (Chan's parallel update). Ignores -1 sentinel."""
    n:         int   = 0
    mean:      float = 0.0
    M2:        float = 0.0
    vmin:      float = np.inf
    vmax:      float = -np.inf
    n_missing: int   = 0
    n_total:   int   = 0

    def update(self, arr: np.ndarray) -> None:
        flat = arr.ravel()
        self.n_total   += len(flat)
        missing         = int((flat == -1).sum())
        self.n_missing += missing
        valid = flat[flat != -1].astype(np.float64)
        if valid.size == 0:
            return
        n_b    = valid.size
        mean_b = float(valid.mean())
        M2_b   = float(valid.var() * n_b)
        delta  = mean_b - self.mean
        n_ab   = self.n + n_b
        self.mean = (self.n * self.mean + n_b * mean_b) / n_ab
        self.M2  += M2_b + delta ** 2 * self.n * n_b / n_ab
        self.n    = n_ab
        self.vmin = min(self.vmin, float(valid.min()))
        self.vmax = max(self.vmax, float(valid.max()))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 0.0

    @property
    def missing_pct(self) -> float:
        return 100.0 * self.n_missing / self.n_total if self.n_total > 0 else np.nan


@dataclass
class HistAccumulator:
    """Fixed-bin histogram without storing raw values."""
    bins:     np.ndarray
    counts:   np.ndarray = field(init=False)
    overflow: int        = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.counts = np.zeros(len(self.bins) - 1, dtype=np.int64)

    def update(self, arr: np.ndarray) -> None:
        valid = arr.ravel()
        valid = valid[valid != -1].astype(np.float64)
        if valid.size == 0:
            return
        lo, hi    = self.bins[0], self.bins[-1]
        in_range  = valid[(valid >= lo) & (valid <= hi)]
        self.overflow += int(valid[(valid < lo) | (valid > hi)].size)
        if in_range.size > 0:
            c, _ = np.histogram(in_range, bins=self.bins)
            self.counts += c


# ── I/O helpers ───────────────────────────────────────────────────────────────

def open_h5_files(paths: list[Path]) -> list[h5py.File]:
    files = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"HDF5 not found: {p}")
        files.append(h5py.File(p, "r"))
    return files


def find_key(files: list[h5py.File], key: str) -> np.ndarray:
    for f in files:
        if key in f:
            return f[key][:]
    raise KeyError(f"'{key}' not found in any HDF5 file")


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def bar_from_histo(ax, h: HistAccumulator, color: str, title: str) -> None:
    mids = (h.bins[:-1] + h.bins[1:]) / 2
    ax.bar(mids, h.counts, width=np.diff(h.bins),
           color=color, edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def stats_table(welford_dict: dict[str, WelfordStats]) -> pd.DataFrame:
    rows = []
    for name, w in welford_dict.items():
        rows.append({
            "feature":    name,
            "n_valid":    w.n,
            "missing_%":  round(w.missing_pct, 2),
            "mean":       round(w.mean, 5),
            "std":        round(w.std,  5),
            "min":        round(w.vmin, 5) if w.n > 0 else np.nan,
            "max":        round(w.vmax, 5) if w.n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ── section 1: dataset overview ───────────────────────────────────────────────

def analyze_dataset(df: pd.DataFrame, figures_dir: Path) -> None:
    print("\n=== 1. Dataset overview ===")

    total = len(df)
    print(f"Total proteins : {total:,}")
    print(f"Labels         : {dict(df['label'].value_counts())}")
    print(f"Sources        : {sorted(df['source'].unique())}")

    src_counts = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    print("\nCounts per source × label:")
    print(src_counts.to_string())

    df = df.copy()
    df["seq_len"] = df["modify_seq"].str.len()
    df["group"]   = df["source"].apply(
        lambda s: "MD/IDR" if s in MD_IDR_SOURCES else "Proteinflow"
    )

    print("\nSequence length by group:")
    for g, sub in df.groupby("group")["seq_len"]:
        print(f"  {g}: n={len(sub):,}  min={sub.min()}  "
              f"median={sub.median():.0f}  mean={sub.mean():.0f}  max={sub.max()}")

    # ── figure 1a: counts per source ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    src_counts.plot(kind="bar", ax=ax)
    ax.set_title("Protein counts per source × split")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Split")
    fig.tight_layout()
    save_fig(fig, figures_dir / "dataset" / "source_counts.png")

    # ── figure 1b: sequence length by group ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (g, sub) in zip(axes, df.groupby("group")):
        ax.hist(sub["seq_len"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.set_title(g)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Count")
        for cutoff, color, lbl in [(256, "orange", "256"), (1024, "red", "1024")]:
            ax.axvline(cutoff, color=color, linestyle="--", linewidth=1, label=lbl)
        ax.legend(fontsize=8)
    fig.suptitle("Sequence length distributions", y=1.02)
    fig.tight_layout()
    save_fig(fig, figures_dir / "dataset" / "seq_length_dist.png")

    # ── figure 1c: CDF ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    lengths = np.sort(df["seq_len"].values)
    ax.plot(lengths, np.arange(1, len(lengths) + 1) / len(lengths), color="#4C72B0")
    for cutoff, color, lbl in [(256, "orange", "256"), (1024, "red", "1024")]:
        pct = (df["seq_len"] <= cutoff).mean()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1,
                   label=f"{lbl} aa  ({100*pct:.1f}%)")
    ax.set_title("Sequence length CDF (all proteins)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Cumulative fraction")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, figures_dir / "dataset" / "seq_length_cdf.png")


# ── section 2: Group A residue features ── (L, 47) ───────────────────────────

def analyze_res_A(df: pd.DataFrame, files: list[h5py.File], figures_dir: Path) -> None:
    sub = df[df["source"].isin(MD_IDR_SOURCES)].reset_index(drop=True)
    print(f"\n=== 2. Group A residue features — MD/IDR  ({len(sub):,} proteins) ===")
    print("     schema: (L, 47)  SASA×2, RMSF, SS×8, chi×12, phi×12, psi×12")

    welford = {k: WelfordStats() for k in RES_A}
    histos  = {
        "sasa_mean": HistAccumulator(np.linspace(-0.05, 5.0,   101)),
        "sasa_std":  HistAccumulator(np.linspace(-0.05, 2.0,   101)),
        "rmsf":      HistAccumulator(np.linspace(-0.5,  50.0,  151)),
        "ss":        HistAccumulator(np.linspace(-0.05, 1.05,  51)),
        "chi":       HistAccumulator(np.linspace(-1.05, 1.05,  101)),
        "phi":       HistAccumulator(np.linspace(-1.05, 1.05,  101)),
        "psi":       HistAccumulator(np.linspace(-1.05, 1.05,  101)),
    }
    ss_counts = np.zeros(8, dtype=np.int64)
    skipped   = 0

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="Group A residue"):
        try:
            rf = find_key(files, f"{row['name']}_res_feature")
        except KeyError:
            skipped += 1
            continue

        if rf.shape[1] != 47:
            print(f"\n  WARNING: {row['name']} has res_feature shape {rf.shape}, expected (L,47) — skipping")
            skipped += 1
            continue

        for feat, cols in RES_A.items():
            chunk = rf[:, cols]
            welford[feat].update(chunk)
            histos[feat].update(chunk)

        # SS dominant class (valid rows only)
        ss = rf[:, 3:11]
        valid = (ss != -1).any(axis=1)
        if valid.any():
            np.add.at(ss_counts, ss[valid].argmax(axis=1), 1)

    print(f"  processed: {len(sub) - skipped:,}  skipped: {skipped}")

    df_stats = stats_table(welford)
    out = figures_dir / "group_A" / "res_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out, index=False)
    print("\n" + df_stats.to_string(index=False))
    print(f"  saved → {out}")

    # figure 2a: missing rates
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_stats["feature"], df_stats["missing_%"], color="#4C72B0")
    ax.set_title("Group A — residue feature missing rates")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing %")
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "res_missing_rates.png")

    # figure 2b: SASA
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bar_from_histo(axes[0], histos["sasa_mean"], "#4C72B0", "SASA mean")
    bar_from_histo(axes[1], histos["sasa_std"],  "#4C72B0", "SASA std")
    fig.suptitle("Group A — SASA distributions", y=1.02)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "sasa_dist.png")

    # figure 2c: RMSF (raw HDF5 values — training normalises by per-protein max)
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_from_histo(ax, histos["rmsf"], "#4C72B0", "RMSF (raw, Å)")
    ax.set_title("Group A — RMSF distribution (raw; training normalises by per-protein max)")
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "rmsf_dist.png")

    # figure 2d: secondary structure class counts
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"SS{i}" for i in range(8)], ss_counts, color="#4C72B0")
    ax.set_title("Group A — Secondary structure class distribution (argmax of 8-class one-hot)")
    ax.set_xlabel("SS class")
    ax.set_ylabel("Residue count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "ss_class_dist.png")

    # figure 2e: torsion angle encodings
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes, ["phi", "psi", "chi"]):
        bar_from_histo(ax, histos[feat], "#DD8452", feat)
    fig.suptitle("Group A — Torsion angle sin/cos encoding distributions", y=1.02)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "torsion_dist.png")


# ── section 3: Group A pairwise features ── (L, L, 10) ───────────────────────

def analyze_pair_A(df: pd.DataFrame, files: list[h5py.File], figures_dir: Path) -> None:
    sub = df[df["source"].isin(MD_IDR_SOURCES)].reset_index(drop=True)
    print(f"\n=== 3. Group A pairwise features — MD/IDR  ({len(sub):,} proteins) ===")
    print("     schema: (L, L, 10)  vdw, hbbb, hbsb, hbss, hp, sb, pc, ps, ts, corr")

    welford           = {k: WelfordStats() for k in PAIR_A}
    pair_bins = {
        "vdw":  np.linspace(-0.5, 30.0, 121),
        "hbbb": np.linspace(-0.5, 10.0,  61),
        "hbsb": np.linspace(-0.5, 10.0,  61),
        "hbss": np.linspace(-0.5, 10.0,  61),
        "hp":   np.linspace(-0.5, 30.0, 121),
        "sb":   np.linspace(-0.5,  5.0,  61),
        "pc":   np.linspace(-0.5,  5.0,  61),
        "ps":   np.linspace(-0.5,  5.0,  61),
        "ts":   np.linspace(-0.5,  5.0,  61),
        "corr": np.linspace(-1.05, 1.05, 101),
    }
    histos            = {k: HistAccumulator(pair_bins[k]) for k in PAIR_A}
    contact_densities = []
    skipped           = 0
    example_name      = None

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="Group A pairwise"):
        try:
            pf = find_key(files, f"{row['name']}_pair_feature")
        except KeyError:
            skipped += 1
            continue

        if pf.shape[2] != 10:
            print(f"\n  WARNING: {row['name']} has pair_feature shape {pf.shape}, expected (L,L,10) — skipping")
            skipped += 1
            continue

        if example_name is None:
            example_name = row["name"]

        valid = pf[:, :, 0] != -1
        contact_densities.append(float(valid.mean()))

        for feat, ch in PAIR_A.items():
            vals = pf[:, :, ch][valid]
            welford[feat].update(vals)
            histos[feat].update(vals)

    print(f"  processed: {len(sub) - skipped:,}  skipped: {skipped}")

    df_stats = stats_table(welford)
    out = figures_dir / "group_A" / "pair_stats.csv"
    df_stats.to_csv(out, index=False)
    print("\n" + df_stats.to_string(index=False))
    print(f"  saved → {out}")

    # figure 3a: contact density
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(contact_densities, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title("Group A — Contact matrix density (fraction of valid pairs)")
    ax.set_xlabel("Density")
    ax.set_ylabel("Protein count")
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "contact_density.png")

    # figure 3b: per-feature distributions (grid)
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = axes.ravel()
    for i, feat in enumerate(PAIR_A):
        bar_from_histo(axes_flat[i], histos[feat], "#4C72B0", feat)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Group A — Pairwise feature distributions (valid pairs only)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "pair_feature_dists.png")

    # figure 3c: fraction non-zero
    fig, ax = plt.subplots(figsize=(9, 4))
    nonzero = [
        100 * (1 - histos[f].counts[0] / max(welford[f].n, 1))
        for f in PAIR_A
    ]
    ax.bar(list(PAIR_A.keys()), nonzero, color="#4C72B0")
    ax.set_title("Group A — Non-zero fraction per pairwise feature")
    ax.set_ylabel("Non-zero %")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_A" / "pair_nonzero_frac.png")

    # figure 3d: example contact maps
    if example_name:
        try:
            pf = find_key(files, f"{example_name}_pair_feature")
            feat_names = list(PAIR_A.keys())
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            for ax, feat in zip(axes.ravel(), feat_names):
                mat = pf[:, :, PAIR_A[feat]].astype(float)
                mat[mat == -1] = np.nan
                im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
                ax.set_title(feat)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for ax in axes.ravel()[len(feat_names):]:
                ax.set_visible(False)
            fig.suptitle(f"Group A — Example contact maps: {example_name}", y=1.01)
            fig.tight_layout()
            save_fig(fig, figures_dir / "group_A" / "example_contact_maps.png")
        except Exception as e:
            print(f"  skipped example contact map: {e}")


# ── section 4: Group B residue features ── (L, 3) NMA only ──────────────────

def analyze_res_B(df: pd.DataFrame, files: list[h5py.File], figures_dir: Path) -> None:
    sub = df[df["source"].isin(PROTEINFLOW_SOURCES)].reset_index(drop=True)
    print(f"\n=== 4. Group B residue features — Proteinflow  ({len(sub):,} proteins) ===")
    print("     schema: (L, 3)  NMA_res1, NMA_res2, NMA_res3")

    welford = {k: WelfordStats() for k in RES_B}
    nma_bins = np.linspace(-5.0, 10.0, 101)
    histos  = {k: HistAccumulator(nma_bins) for k in RES_B}
    skipped = 0

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="Group B residue"):
        try:
            rf = find_key(files, f"{row['name']}_res_feature")
        except KeyError:
            skipped += 1
            continue

        if rf.shape[1] != 3:
            print(f"\n  WARNING: {row['name']} has res_feature shape {rf.shape}, expected (L,3) — skipping")
            skipped += 1
            continue

        for feat, col in RES_B.items():
            vals = rf[:, col]
            welford[feat].update(vals)
            histos[feat].update(vals)

    print(f"  processed: {len(sub) - skipped:,}  skipped: {skipped}")

    df_stats = stats_table(welford)
    out = figures_dir / "group_B" / "nma_res_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out, index=False)
    print("\n" + df_stats.to_string(index=False))
    print(f"  saved → {out}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes, RES_B):
        bar_from_histo(ax, histos[feat], "#55A868", feat)
    fig.suptitle("Group B — NMA residue mode distributions (Proteinflow)", y=1.02)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_B" / "nma_res_dist.png")


# ── section 5: Group B pairwise features ── (L, L, 3) NMA only ───────────────

def analyze_pair_B(df: pd.DataFrame, files: list[h5py.File], figures_dir: Path) -> None:
    sub = df[df["source"].isin(PROTEINFLOW_SOURCES)].reset_index(drop=True)
    print(f"\n=== 5. Group B pairwise features — Proteinflow  ({len(sub):,} proteins) ===")
    print("     schema: (L, L, 3)  NMA_pair1, NMA_pair2, NMA_pair3")

    welford           = {k: WelfordStats() for k in PAIR_B}
    nma_bins          = np.linspace(-5.0, 10.0, 101)
    histos            = {k: HistAccumulator(nma_bins) for k in PAIR_B}
    contact_densities = []
    skipped           = 0
    example_name      = None

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="Group B pairwise"):
        try:
            pf = find_key(files, f"{row['name']}_pair_feature")
        except KeyError:
            skipped += 1
            continue

        if pf.shape[2] != 3:
            print(f"\n  WARNING: {row['name']} has pair_feature shape {pf.shape}, expected (L,L,3) — skipping")
            skipped += 1
            continue

        if example_name is None:
            example_name = row["name"]

        valid = pf[:, :, 0] != -1
        contact_densities.append(float(valid.mean()))

        for feat, ch in PAIR_B.items():
            vals = pf[:, :, ch][valid]
            welford[feat].update(vals)
            histos[feat].update(vals)

    print(f"  processed: {len(sub) - skipped:,}  skipped: {skipped}")

    df_stats = stats_table(welford)
    out = figures_dir / "group_B" / "nma_pair_stats.csv"
    df_stats.to_csv(out, index=False)
    print("\n" + df_stats.to_string(index=False))
    print(f"  saved → {out}")

    # figure 5a: NMA pairwise density
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(contact_densities, bins=60, color="#55A868", edgecolor="white", linewidth=0.4)
    ax.set_title("Group B — NMA pairwise density (fraction of valid pairs)")
    ax.set_xlabel("Density")
    ax.set_ylabel("Protein count")
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_B" / "nma_pair_density.png")

    # figure 5b: NMA pairwise distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes, PAIR_B):
        bar_from_histo(ax, histos[feat], "#55A868", feat)
    fig.suptitle("Group B — NMA pairwise mode distributions (Proteinflow)", y=1.02)
    fig.tight_layout()
    save_fig(fig, figures_dir / "group_B" / "nma_pair_dist.png")

    # figure 5c: example NMA contact maps
    if example_name:
        try:
            pf = find_key(files, f"{example_name}_pair_feature")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, feat in zip(axes, PAIR_B):
                mat = pf[:, :, PAIR_B[feat]].astype(float)
                mat[mat == -1] = np.nan
                im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
                ax.set_title(feat)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Group B — Example NMA maps: {example_name}", y=1.01)
            fig.tight_layout()
            save_fig(fig, figures_dir / "group_B" / "example_nma_maps.png")
        except Exception as e:
            print(f"  skipped example NMA map: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for DynamicsPLM dataset")
    parser.add_argument("--data_dir", default=str(DATA_DIR),
                        help=f"Training data directory (default: {DATA_DIR})")
    parser.add_argument("--figures_dir", default="save_dir/figures",
                        help="Output directory for figures (default: save_dir/figures/)")
    parser.add_argument("--split", default="all", choices=["all", "train", "test"],
                        help="CSV label split to analyse (default: all)")
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
    figures_dir = Path(args.figures_dir)
    data_dir    = Path(args.data_dir)
    CSV_PATH    = data_dir / config["file_path"]["train_df"]
    H5_PATHS    = [data_dir / name for name in config["file_path"]["h5_files"]]

    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if args.split != "all":
        df = df[df["label"] == args.split].reset_index(drop=True)
    print(f"Rows loaded : {len(df):,}  (split={args.split})")

    print("\nOpening HDF5 files:")
    files = open_h5_files(H5_PATHS)
    for p, f in zip(H5_PATHS, files):
        print(f"  {p.name}  —  {len(f):,} datasets")

    analyze_dataset(df, figures_dir)
    analyze_res_A(df, files, figures_dir)
    analyze_pair_A(df, files, figures_dir)
    analyze_res_B(df, files, figures_dir)
    analyze_pair_B(df, files, figures_dir)

    for f in files:
        f.close()

    print(f"\nDone. All outputs saved under {figures_dir}/")


if __name__ == "__main__":
    main()
