"""
tests/integration/test_pipeline.py
Integration tests: verify that clean → enrich → preprocess produces
correctly-shaped, non-leaking, logically valid outputs.
"""
import sys, os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/data", "src/features"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean import clean_dataset
    from enrich import build_output, SEASON_SIZE, SEASON_COUNT
    from preprocess import (
        validate_input as pp_validate,
        split_data,
        encode_features,
        scale_features,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        DATE_COLUMN,
        _TRAIN_END,
        _VAL_END,
    )
    _IMPORT_OK = True
except ModuleNotFoundError as e:
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason=f"Import failed: {_IMPORT_OK}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_season(season_idx: int = 0) -> pd.DataFrame:
    """Create one season (380 rows) of realistic raw match data."""
    n = SEASON_SIZE
    rng = np.random.default_rng(season_idx)
    teams = [f"Team{i:02d}" for i in range(20)]
    results = rng.choice(["H", "D", "A"], size=n)

    return pd.DataFrame({
        "Date":     pd.date_range(
            f"{2015 + season_idx}-08-08", periods=n, freq="D"
        ).strftime("%d/%m/%Y"),
        "HomeTeam": [teams[i % 20] for i in range(n)],
        "AwayTeam": [teams[(i + 1) % 20] for i in range(n)],
        "FTHG":     rng.integers(0, 5, size=n),
        "FTAG":     rng.integers(0, 5, size=n),
        "FTR":      results,
        "Referee":  [f"Ref{i % 20}" for i in range(n)],
        "HS":       rng.integers(3, 25, size=n),
        "AS":       rng.integers(3, 25, size=n),
        "HST":      rng.integers(1, 10, size=n),
        "AST":      rng.integers(1, 10, size=n),
        "HF":       rng.integers(5, 20, size=n),
        "AF":       rng.integers(5, 20, size=n),
        "HC":       rng.integers(1, 12, size=n),
        "AC":       rng.integers(1, 12, size=n),
        "HY":       rng.integers(0, 5,  size=n),
        "AY":       rng.integers(0, 5,  size=n),
        "HR":       rng.integers(0, 2,  size=n),
        "AR":       rng.integers(0, 2,  size=n),
    })


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Full 10-season raw dataset (3800 rows)."""
    seasons = [_make_raw_season(i) for i in range(SEASON_COUNT)]
    return pd.concat(seasons, ignore_index=True)


@pytest.fixture(scope="module")
def cleaned_csv(tmp_path_factory, raw_df) -> Path:
    d = tmp_path_factory.mktemp("pipeline")
    raw_path = d / "raw.csv"
    clean_path = d / "clean.csv"
    raw_df.to_csv(raw_path, index=False)
    clean_dataset(str(raw_path), str(clean_path))
    return clean_path


@pytest.fixture(scope="module")
def enriched_df(cleaned_csv) -> pd.DataFrame:
    df = pd.read_csv(cleaned_csv)
    return build_output(df)


# ---------------------------------------------------------------------------
# Stage 1: clean_dataset
# ---------------------------------------------------------------------------

class TestCleanStage:
    def test_output_file_created(self, cleaned_csv):
        assert cleaned_csv.exists()

    def test_ast_never_exceeds_as(self, cleaned_csv):
        df = pd.read_csv(cleaned_csv)
        assert (df["AST"] <= df["AS"]).all()

    def test_row_count_unchanged(self, raw_df, cleaned_csv):
        df = pd.read_csv(cleaned_csv)
        assert len(df) == len(raw_df)

    def test_dates_all_in_ddmmyyyy(self, cleaned_csv):
        df = pd.read_csv(cleaned_csv)
        pattern = r"^\d{2}/\d{2}/\d{4}$"
        assert df["Date"].str.match(pattern).all()


# ---------------------------------------------------------------------------
# Stage 2: enrich / build_output
# ---------------------------------------------------------------------------

class TestEnrichStage:
    def test_output_has_expected_row_count(self, enriched_df, raw_df):
        assert len(enriched_df) == len(raw_df)

    def test_no_nulls_in_enriched_features(self, enriched_df):
        feat_cols = [c for c in NUMERICAL_FEATURES if c in enriched_df.columns]
        assert enriched_df[feat_cols].isnull().sum().sum() == 0

    def test_ftr_values_valid(self, enriched_df):
        assert set(enriched_df[TARGET_COLUMN].unique()).issubset({"H", "D", "A"})

    def test_hpos_in_valid_range(self, enriched_df):
        # League has 20 teams; positions 1–20
        assert enriched_df["HPOS"].between(1, 20).all()
        assert enriched_df["APOS"].between(1, 20).all()

    def test_goals_scored_nonnegative(self, enriched_df):
        for col in ["HGS", "AGS", "HGSH", "AGSA"]:
            assert (enriched_df[col] >= 0).all(), f"{col} has negative values"

    def test_points_per_game_bounded(self, enriched_df):
        # PPG can't exceed 3 (max points per match)
        for col in ["HPPG", "APPG", "HPPGH", "APPGA"]:
            assert (enriched_df[col] <= 3.0 + 1e-9).all(), f"{col} > 3"

    def test_clean_sheets_rate_bounded(self, enriched_df):
        for col in ["HCS", "ACS"]:
            assert (enriched_df[col] >= 0).all()
            assert (enriched_df[col] <= 1.0 + 1e-9).all(), f"{col} > 1"

    def test_required_columns_present(self, enriched_df):
        missing = [c for c in NUMERICAL_FEATURES if c not in enriched_df.columns]
        assert not missing, f"Missing enriched columns: {missing}"


# ---------------------------------------------------------------------------
# Stage 3: preprocess — split + encode + scale
# ---------------------------------------------------------------------------

class TestPreprocessStage:
    def test_validate_passes_on_enriched(self, enriched_df):
        pp_validate(enriched_df)   # must not raise

    def test_split_sizes(self, enriched_df):
        train, val, test = split_data(enriched_df)
        assert len(train) == _TRAIN_END
        assert len(val)   == _VAL_END - _TRAIN_END
        assert len(test)  == len(enriched_df) - _VAL_END

    def test_no_temporal_leakage(self, enriched_df):
        """Train must contain only rows that come before val and test."""
        df = enriched_df.copy()
        df["_idx"] = range(len(df))
        train, val, test = split_data(df)
        assert train["_idx"].max() < val["_idx"].min()
        assert val["_idx"].max()   < test["_idx"].min()

    def test_encoded_target_is_numeric(self, enriched_df):
        train, val, test = split_data(enriched_df)
        enc_train, enc_val, enc_test, _ = encode_features(train, val, test)
        for split in (enc_train, enc_val, enc_test):
            assert split[TARGET_COLUMN].dtype in [int, np.int64, np.int32, float, np.float64]

    def test_scaled_train_mean_near_zero(self, enriched_df):
        train, val, test = split_data(enriched_df)
        train, val, test, _ = encode_features(train, val, test)
        X_train = train.drop(columns=[TARGET_COLUMN])
        X_val   = val.drop(columns=[TARGET_COLUMN])
        X_test  = test.drop(columns=[TARGET_COLUMN])
        num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        X_train_s, _, _, _ = scale_features(X_train, X_val, X_test, num_cols)
        means = X_train_s[num_cols].mean()
        assert (means.abs() < 1e-6).all()

    def test_date_not_in_final_features(self, enriched_df):
        train, val, test = split_data(enriched_df)
        enc_train, enc_val, enc_test, _ = encode_features(train, val, test)
        for split in (enc_train, enc_val, enc_test):
            assert DATE_COLUMN not in split.columns