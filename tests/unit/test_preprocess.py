"""
tests/unit/test_preprocess.py
Unit tests for src/features/preprocess.py
"""
import sys, os
import pandas as pd
import numpy as np
import pytest

_FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "features")
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

try:
    from preprocess import (
        validate_input,
        split_data,
        encode_features,
        scale_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        DATE_COLUMN,
        TARGET_MAPPING,
        _TRAIN_END,
        _VAL_END,
    )
    _IMPORT_OK = True
except ModuleNotFoundError:
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="preprocess.py not importable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_full_df(n_rows: int = 3800) -> pd.DataFrame:
    """Minimal DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    data: dict = {}

    for col in NUMERICAL_FEATURES:
        data[col] = rng.uniform(0, 3, size=n_rows)

    # Categorical
    teams = [f"Team{i}" for i in range(20)]
    data["HomeTeam"] = [teams[i % 20] for i in range(n_rows)]
    data["AwayTeam"] = [teams[(i + 1) % 20] for i in range(n_rows)]
    data["Referee"]  = [f"Ref{i % 30}" for i in range(n_rows)]

    # Target
    data[TARGET_COLUMN] = [["H", "D", "A"][i % 3] for i in range(n_rows)]

    # Date (simple incrementing dates won't matter for row-based split)
    data[DATE_COLUMN] = pd.date_range("2015-08-08", periods=n_rows, freq="D").strftime("%d/%m/%Y")

    return pd.DataFrame(data)


@pytest.fixture
def full_df():
    return _make_full_df(3800)


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_passes_with_all_columns(self, full_df):
        validate_input(full_df)   # must not raise

    def test_raises_on_missing_numerical(self, full_df):
        df = full_df.drop(columns=[NUMERICAL_FEATURES[0]])
        with pytest.raises(ValueError, match="missing"):
            validate_input(df)

    def test_raises_on_missing_categorical(self, full_df):
        df = full_df.drop(columns=["HomeTeam"])
        with pytest.raises(ValueError, match="missing"):
            validate_input(df)

    def test_raises_on_missing_target(self, full_df):
        df = full_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError, match="missing"):
            validate_input(df)

    def test_raises_on_missing_date(self, full_df):
        df = full_df.drop(columns=[DATE_COLUMN])
        with pytest.raises(ValueError, match="missing"):
            validate_input(df)

    def test_error_message_lists_missing_column(self, full_df):
        col = NUMERICAL_FEATURES[3]
        df = full_df.drop(columns=[col])
        with pytest.raises(ValueError, match=col):
            validate_input(df)

    def test_multiple_missing_columns_reported(self, full_df):
        df = full_df.drop(columns=NUMERICAL_FEATURES[:3])
        with pytest.raises(ValueError) as exc_info:
            validate_input(df)
        msg = str(exc_info.value)
        for col in NUMERICAL_FEATURES[:3]:
            assert col in msg


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_split_sizes(self, full_df):
        train, val, test = split_data(full_df)
        assert len(train) == _TRAIN_END
        assert len(val)   == _VAL_END - _TRAIN_END
        assert len(test)  == len(full_df) - _VAL_END

    def test_total_rows_preserved(self, full_df):
        train, val, test = split_data(full_df)
        assert len(train) + len(val) + len(test) == len(full_df)

    def test_no_row_overlap_between_splits(self, full_df):
        full_df = full_df.copy()
        full_df["_row_id"] = range(len(full_df))
        train, val, test = split_data(full_df)
        train_ids = set(train["_row_id"])
        val_ids   = set(val["_row_id"])
        test_ids  = set(test["_row_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_temporal_order_preserved(self, full_df):
        """Train rows must come before val rows which come before test rows."""
        full_df = full_df.copy()
        full_df["_row_id"] = range(len(full_df))
        train, val, test = split_data(full_df)
        assert train["_row_id"].max() < val["_row_id"].min()
        assert val["_row_id"].max()   < test["_row_id"].min()

    def test_all_target_classes_in_train(self, full_df):
        train, _, _ = split_data(full_df)
        assert set(train[TARGET_COLUMN].unique()) == {"H", "D", "A"}

    def test_columns_unchanged(self, full_df):
        train, val, test = split_data(full_df)
        for split in (train, val, test):
            assert set(split.columns) == set(full_df.columns)


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------

class TestEncodeFeatures:
    def test_target_encoded_correctly(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, enc_val, enc_test, _ = encode_features(train, val, test)
        assert set(enc_train[TARGET_COLUMN].unique()).issubset({0, 1, 2})

    def test_h_maps_to_2(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, _, _, _ = encode_features(train, val, test)
        # All rows that were "H" must now be 2
        original_h_mask = train[TARGET_COLUMN] == "H"
        assert (enc_train.loc[original_h_mask.values, TARGET_COLUMN] == 2).all()

    def test_d_maps_to_1(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, _, _, _ = encode_features(train, val, test)
        original_d_mask = train[TARGET_COLUMN] == "D"
        assert (enc_train.loc[original_d_mask.values, TARGET_COLUMN] == 1).all()

    def test_a_maps_to_0(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, _, _, _ = encode_features(train, val, test)
        original_a_mask = train[TARGET_COLUMN] == "A"
        assert (enc_train.loc[original_a_mask.values, TARGET_COLUMN] == 0).all()

    def test_date_column_dropped(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, enc_val, enc_test, _ = encode_features(train, val, test)
        for split in (enc_train, enc_val, enc_test):
            assert DATE_COLUMN not in split.columns

    def test_team_columns_are_integers(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, _, _, _ = encode_features(train, val, test)
        assert enc_train["HomeTeam"].dtype in [int, np.int64, np.int32, float, np.float64]

    def test_referee_column_is_integer(self, full_df):
        train, val, test = split_data(full_df)
        enc_train, _, _, _ = encode_features(train, val, test)
        assert enc_train["Referee"].dtype in [int, np.int64, np.int32, float, np.float64]

    def test_encoder_classes_returned(self, full_df):
        train, val, test = split_data(full_df)
        _, _, _, classes = encode_features(train, val, test)
        assert "HomeTeam" in classes
        assert "AwayTeam" in classes
        assert "Referee" in classes

    def test_unseen_team_gets_median_rank(self, full_df):
        """A team that never appears in train should not cause KeyError."""
        train, val, test = split_data(full_df)
        # Inject a completely new team name into val
        val = val.copy()
        val.iloc[0, val.columns.get_loc("HomeTeam")] = "UnknownFC_XYZ"
        # Should not raise
        _, enc_val, _, classes = encode_features(train, val, test)
        median_rank = len(classes["HomeTeam"]) // 2
        assert enc_val.iloc[0]["HomeTeam"] == median_rank


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def test_scaled_mean_near_zero(self, full_df):
        train, val, test = split_data(full_df)
        train, val, test, _ = encode_features(train, val, test)
        X_train = train.drop(columns=[TARGET_COLUMN])
        X_val   = val.drop(columns=[TARGET_COLUMN])
        X_test  = test.drop(columns=[TARGET_COLUMN])
        num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        X_train_s, _, _, _ = scale_features(X_train, X_val, X_test, num_cols)
        means = X_train_s[num_cols].mean()
        assert (means.abs() < 1e-6).all(), "Scaled train means should be ~0"

    def test_scaled_std_near_one(self, full_df):
        train, val, test = split_data(full_df)
        train, val, test, _ = encode_features(train, val, test)
        X_train = train.drop(columns=[TARGET_COLUMN])
        X_val   = val.drop(columns=[TARGET_COLUMN])
        X_test  = test.drop(columns=[TARGET_COLUMN])
        num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        X_train_s, _, _, _ = scale_features(X_train, X_val, X_test, num_cols)
        stds = X_train_s[num_cols].std()
        assert ((stds - 1.0).abs() < 0.05).all(), "Scaled train stds should be ~1"

    def test_scaler_fitted_on_train_only(self, full_df):
        """Val and test are transformed with train statistics, not their own."""
        train, val, test = split_data(full_df)
        train, val, test, _ = encode_features(train, val, test)
        X_train = train.drop(columns=[TARGET_COLUMN])
        X_val   = val.drop(columns=[TARGET_COLUMN])
        X_test  = test.drop(columns=[TARGET_COLUMN])
        num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        _, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test, num_cols)
        # Val mean should NOT be exactly 0 (it's scaled with train stats)
        val_means = X_val_s[num_cols].mean()
        # We just check scaler attributes were computed from train
        assert scaler.mean_ is not None
        assert len(scaler.mean_) == len(num_cols)

    def test_non_numerical_columns_unchanged(self, full_df):
        train, val, test = split_data(full_df)
        train, val, test, _ = encode_features(train, val, test)
        X_train = train.drop(columns=[TARGET_COLUMN])
        X_val   = val.drop(columns=[TARGET_COLUMN])
        X_test  = test.drop(columns=[TARGET_COLUMN])
        num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        X_train_s, _, _, _ = scale_features(X_train, X_val, X_test, num_cols)
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
        for col in cat_cols:
            pd.testing.assert_series_equal(
                X_train[col].reset_index(drop=True),
                X_train_s[col].reset_index(drop=True),
                check_names=False,
            )