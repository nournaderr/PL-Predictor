from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERICAL_FEATURES: list[str] = [
    "HPPG", "HPPGH", "HPPG_FORM", "HPPGH_FORM",
    "APPG", "APPGA", "APPG_FORM", "APPGA_FORM",
    "HGS", "HGSH", "HGS_FORM", "HGSH_FORM",
    "AGS", "AGSA", "AGS_FORM", "AGSA_FORM",
    "HGC", "HGCH", "HGC_FORM", "HGCH_FORM",
    "AGC", "AGCA", "AGC_FORM", "AGCA_FORM",
    "HCS", "HCSH", "HCS_FORM", "HCSH_FORM",
    "ACS", "ACSA", "ACS_FORM", "ACSA_FORM",
    "HPOS", "APOS", "HPDU", "HPDD", "APDU", "APDD",
    "HS_FORM", "AS_FORM", "HST_FORM", "AST_FORM",
    "HF_FORM", "AF_FORM", "HC_FORM", "AC_FORM",
    "HY_FORM", "AY_FORM", "HR_FORM", "AR_FORM",
]

CATEGORICAL_FEATURES: list[str] = ["HomeTeam", "AwayTeam", "Referee"]
TARGET_COLUMN: str = "FTR"
DATE_COLUMN: str = "Date"
TARGET_MAPPING: dict[str, int] = {"H": 2, "D": 1, "A": 0}

REQUIRED_COLUMNS: list[str] = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN, DATE_COLUMN]
)

# Row boundaries for the time-aware split (upper bounds are exclusive)
_TRAIN_END: int = 3040   # rows 0–3039   → seasons 2015–2022
_VAL_END: int = 3420     # rows 3040–3419 → season 2023

def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError listing every missing required column before processing begins."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataset is missing {len(missing)} required column(s):\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train, val, test) using a fixed time-aware row split with no shuffling.
    No shuffling is performed to prevent temporal data leakage.
    """
    train = df.iloc[:_TRAIN_END].copy()
    val = df.iloc[_TRAIN_END:_VAL_END].copy()
    test = df.iloc[_VAL_END:].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        total = len(split)
        counts = split[TARGET_COLUMN].value_counts()
        print(f"\n  {name} ({total:,} rows):")
        for label in ("H", "D", "A"):
            n = int(counts.get(label, 0))
            print(f"    {label}: {n:>5}  ({n / total * 100:5.1f}%)")

    return train, val, test


def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list]]:
    """
    Encode the target and categorical columns; drop the Date column.

    FTR is mapped via TARGET_MAPPING (H→2, D→1, A→0).

    HomeTeam and AwayTeam are ordinally encoded by cumulative wins in the
    training set — teams with more historical wins receive higher integers.
    This creates a meaningful ordinal relationship (stronger team = higher
    value) that tree-based models can exploit via threshold splits, unlike
    alphabetical LabelEncoding which implies an arbitrary order.

    Referee is ordinally encoded by matches officiated in training data —
    more experienced referees receive higher integers.

    Encoding is fitted ONLY on training data, then applied to val and test.
    Unseen labels (promoted teams, new referees) receive -1 with a warning.

    Returns new DataFrames (not modified in-place) and a dict mapping each
    categorical column name to the ordered list of classes (low→high rank)
    for metadata.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Encode target
    for split in (train, val, test):
        split[TARGET_COLUMN] = split[TARGET_COLUMN].map(TARGET_MAPPING)

    encoder_classes: dict[str, list] = {}

    # ── HomeTeam & AwayTeam: rank by total wins in training data ──────────
    home_wins = (
        train[train["HomeTeam"].notna()]
        .groupby("HomeTeam")
        .apply(lambda g: (g[TARGET_COLUMN] == TARGET_MAPPING["H"]).sum())
    )
    away_wins = (
        train[train["AwayTeam"].notna()]
        .groupby("AwayTeam")
        .apply(lambda g: (g[TARGET_COLUMN] == TARGET_MAPPING["A"]).sum())
    )
    all_teams = sorted(set(home_wins.index) | set(away_wins.index))
    total_wins = {
        team: int(home_wins.get(team, 0)) + int(away_wins.get(team, 0))
        for team in all_teams
    }
    # Sort ascending by wins so highest-win team gets the highest integer
    teams_ranked = sorted(all_teams, key=lambda t: total_wins[t])
    team_rank_map: dict[str, int] = {
        team: rank for rank, team in enumerate(teams_ranked)
    }
    encoder_classes["HomeTeam"] = teams_ranked
    encoder_classes["AwayTeam"] = teams_ranked

    for col in ("HomeTeam", "AwayTeam"):
        for split_name, split in [("Train", train), ("Val", val), ("Test", test)]:
            unseen = sorted(set(split[col].dropna()) - set(team_rank_map))
            if unseen and split_name != "Train":
                print(
                    f"  WARNING [{split_name}] '{col}' contains "
                    f"{len(unseen)} unseen label(s): {unseen}. Assigning -1."
                )
            split[col] = split[col].map(lambda v, m=team_rank_map: m.get(v, -1))

    # ── Referee: rank by matches officiated in training data ──────────────
    ref_counts = train["Referee"].value_counts()
    refs_ranked = ref_counts.index[::-1].tolist()  # least→most experienced
    ref_rank_map: dict[str, int] = {
        ref: rank for rank, ref in enumerate(refs_ranked)
    }
    encoder_classes["Referee"] = refs_ranked

    for split_name, split in [("Train", train), ("Val", val), ("Test", test)]:
        unseen = sorted(set(split["Referee"].dropna()) - set(ref_rank_map))
        if unseen and split_name != "Train":
            print(
                f"  WARNING [{split_name}] 'Referee' contains "
                f"{len(unseen)} unseen label(s): {unseen}. Assigning -1."
            )
        split["Referee"] = split["Referee"].map(
            lambda v, m=ref_rank_map: m.get(v, -1)
        )

    # Drop Date
    train = train.drop(columns=[DATE_COLUMN])
    val = val.drop(columns=[DATE_COLUMN])
    test = test.drop(columns=[DATE_COLUMN])

    # Print encoding summary
    top3_teams = teams_ranked[-3:]
    bot3_teams = teams_ranked[:3]
    print(f"\n  {'Column':<15} {'Method':<25} {'Details'}")
    print("  " + "-" * 75)
    print(f"  {'FTR':<15} {'Manual mapping':<25} H→2, D→1, A→0")
    print(f"  {'Date':<15} {'Dropped':<25} Not needed after temporal split")
    print(f"  {'HomeTeam':<15} {'Ordinal (wins)':<25} "
          f"weakest→0: {bot3_teams}  …  strongest→{len(teams_ranked)-1}: {top3_teams}")
    print(f"  {'AwayTeam':<15} {'Ordinal (wins)':<25} same mapping as HomeTeam")
    print(f"  {'Referee':<15} {'Ordinal (matches)':<25} "
          f"least experienced→0, most→{len(refs_ranked)-1}  ({len(refs_ranked)} referees)")
    print(f"\n  Numerical features ({len(NUMERICAL_FEATURES)}): StandardScaler — applied in SCALING step")

    return train, val, test, encoder_classes

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on training numerical columns; apply to val and test.

    Encoded categorical columns and the target are intentionally excluded.
    Returns copies of the three updated feature DataFrames and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_val, X_test, scaler


def select_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Two-step feature selection computed on training data; same mask applied to val/test.

    Pearson correlation is computed ONLY on numerical features — categorical
    features (HomeTeam, AwayTeam, Referee) are excluded from both filter steps
    and always kept, because Pearson correlation on label-encoded nominals is
    statistically meaningless (the integer assignment is alphabetically arbitrary).

    Step A — low-correlation filter:
        Drop any numerical feature where |Pearson r with encoded target| < 0.03.

    Step B — near-duplicate filter:
        For every remaining numerical pair with |inter-feature r| > 0.97, drop
        the member with the lower |r with target| (ties broken by column order).

    Prints a formatted table of every dropped feature, its reason, and its
    Pearson r value.  Returns the pruned splits and a dropped-features dict
    ready for JSON serialisation.
    """
    # Only evaluate numerical features — exclude encoded categoricals
    numerical_in_X = [c for c in NUMERICAL_FEATURES if c in X_train.columns]

    target_corr: dict[str, float] = {
        col: float(X_train[col].corr(y_train)) for col in numerical_in_X
    }
    abs_target_corr: dict[str, float] = {k: abs(v) for k, v in target_corr.items()}

    dropped: dict[str, dict[str, Any]] = {}

    # Step A — low correlation with target
    for col, ac in abs_target_corr.items():
        if ac < 0.03:
            dropped[col] = {
                "reason": "low_correlation_with_target",
                "correlation": target_corr[col],
            }

    # Step B — near-duplicate pairs (greedy, evaluated on Step-A survivors)
    remaining: list[str] = [c for c in numerical_in_X if c not in dropped]
    inter_corr = X_train[remaining].corr().abs()
    already_dropped: set[str] = set()

    for i, c1 in enumerate(remaining):
        for c2 in remaining[i + 1:]:
            if c1 in already_dropped or c2 in already_dropped:
                continue
            if inter_corr.loc[c1, c2] > 0.97:
                loser = c1 if abs_target_corr[c1] <= abs_target_corr[c2] else c2
                already_dropped.add(loser)
                dropped[loser] = {
                    "reason": "near_duplicate_with_higher_corr_feature",
                    "correlation": target_corr[loser],
                }
    if dropped:
        print(f"\n  {'Feature':<25} {'Reason':<45} {'Corr':>8}")
        print("  " + "-" * 80)
        for feat, info in dropped.items():
            print(
                f"  {feat:<25} {info['reason']:<45} {info['correlation']:>8.4f}"
            )

    n_remain = len(X_train.columns) - len(dropped)
    print(
        f"\n  Dropped {len(dropped)} feature(s).  "
        f"{n_remain} feature(s) remain."
    )

    # Categorical features are always kept; only drop from numerical set
    keep: list[str] = [c for c in X_train.columns if c not in dropped]
    return X_train[keep].copy(), X_val[keep].copy(), X_test[keep].copy(), dropped


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample minority class(es) in the training set using SMOTE (random_state=42).

    Prints class distribution before and after resampling.
    Only called when the --smote CLI flag is supplied.

    NOTE: By default this pipeline does NOT apply SMOTE. The mild class imbalance
    (H: 44.5%, D: 23.3%, A: 32.2%, ratio ~1.9:1) is handled at training time via
    class_weight='balanced' on each model. Pass --smote only to run the explicit
    oversampling experiment and compare Draw F1 against the default strategy.

    FIX: both returned objects have their index reset so they remain aligned
    after SMOTE introduces new synthetic rows.
    """
    from imblearn.over_sampling import SMOTE  # lazy import; guarded by CLI flag

    def _print_dist(y: pd.Series, label: str) -> None:
        total = len(y)
        print(f"  {label}:")
        for encoded, name in [(2, "H"), (1, "D"), (0, "A")]:
            n = int((y == encoded).sum())
            print(f"    {name} ({encoded}): {n:>6}  ({n / total * 100:5.1f}%)")

    _print_dist(y_train, "Before")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train.values, y_train.values)

    # FIX: reset_index ensures X and y stay aligned if any downstream code
    # joins them by index after SMOTE has introduced new synthetic rows
    X_res_df = pd.DataFrame(X_res, columns=X_train.columns).reset_index(drop=True)
    y_res_s = pd.Series(y_res, name=TARGET_COLUMN).reset_index(drop=True)

    _print_dist(y_res_s, "After")

    return X_res_df, y_res_s


def save_outputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    all_numerical_cols: list[str],
    encoder_classes: dict[str, list],
    dropped_features: dict[str, dict[str, Any]],
    smote_applied: bool,
    output_dir: Path,
) -> None:
    """
    Write the six CSV splits and preprocessing_metadata.json to output_dir.

    CSV files: X_train, X_val, X_test, y_train, y_val, y_test (no row index).

    The scaler was fitted on all_numerical_cols (pre-selection). After feature
    selection some of those columns may have been dropped, so scaler_mean and
    scaler_scale in metadata are filtered to only the numerical columns that
    survived selection — keeping them aligned with feature_names so downstream
    code (e.g. a Streamlit predictor) can correctly scale new inputs.

    FIX: scaler_mean and scaler_scale are now filtered to surviving numerical
    features only, preventing a length mismatch with feature_names.

    Metadata fields written to preprocessing_metadata.json:
        feature_names         — ordered feature list after selection
        dropped_features      — {name: {reason, correlation}}
        class_mapping         — {"H":2, "D":1, "A":0}
        split_sizes           — {"train":N, "val":N, "test":N}
        scaled_features       — numerical columns the scaler covers (post-selection)
        scaler_mean           — list of floats aligned to scaled_features
        scaler_scale          — list of floats aligned to scaled_features
        label_encoder_classes — {col: [class, ...]}
        smote_applied         — bool
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag, obj in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        obj.to_csv(output_dir / f"{tag}.csv", index=False)

    surviving_numerical = [c for c in all_numerical_cols if c in X_train.columns]
    surviving_idx = [all_numerical_cols.index(c) for c in surviving_numerical]

    metadata: dict[str, Any] = {
        "feature_names": X_train.columns.tolist(),
        "dropped_features": dropped_features,
        "class_mapping": TARGET_MAPPING,
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "scaled_features": surviving_numerical,
        "scaler_mean": [scaler.mean_[i] for i in surviving_idx],
        "scaler_scale": [scaler.scale_[i] for i in surviving_idx],
        "label_encoder_classes": encoder_classes,
        "smote_applied": smote_applied,
    }
    with open(output_dir / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Written to {output_dir}/")
    for tag in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        print(f"    {tag}.csv")
    print("    preprocessing_metadata.json")


def main() -> None:
    """Parse CLI arguments and execute the full preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the enriched PL dataset for modelling. "
            "Performs time-aware splitting, encoding, standard scaling, "
            "Pearson-based feature selection, and optional SMOTE balancing."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the enriched dataset CSV (output of enrich.py).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/processed"),
        type=Path,
        help="Directory for output CSV files and metadata (default: data/processed/).",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help=(
            "Apply SMOTE to the training set after feature selection. "
            "Off by default — imbalance is handled via class_weight='balanced' "
            "during model training instead. Use this flag only to run the "
            "explicit oversampling experiment."
        ),
    )
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    validate_input(df)

    print("\nSPLIT")
    train, val, test = split_data(df)

    print("\nENCODING")
    train, val, test, encoder_classes = encode_features(train, val, test)

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN].rename(TARGET_COLUMN)
    X_val = val.drop(columns=[TARGET_COLUMN])
    y_val = val[TARGET_COLUMN].rename(TARGET_COLUMN)
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].rename(TARGET_COLUMN)

    print("\nSCALING")
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test, numerical_cols
    )
    print(f"  Scaled {len(numerical_cols)} numerical feature(s).")

    print("\nFEATURE SELECTION")
    X_train, X_val, X_test, dropped_features = select_features(
        X_train, X_val, X_test, y_train
    )

    smote_applied = False
    if args.smote:
        print("\nSMOTE")
        X_train, y_train = apply_smote(X_train, y_train)
        smote_applied = True

    print("\nSAVE")
    save_outputs(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler,
        numerical_cols,       # full pre-selection list; save_outputs filters internally
        encoder_classes,
        dropped_features,
        smote_applied,
        args.output_dir,
    )


if __name__ == "__main__":
    main()