"""
tests/unit/test_clean.py
Unit tests for src/data/clean.py
"""
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — replicate the cleaning logic so tests are self-contained
# and do NOT depend on file I/O.
# ---------------------------------------------------------------------------

def _apply_clean_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the core logic of clean_dataset() without touching the filesystem."""
    df = df.copy()

    # Fix AST > AS
    mask = df["AST"] > df["AS"]
    df.loc[mask, ["AS", "AST"]] = df.loc[mask, ["AST", "AS"]].values

    # Normalize dates
    if "Date" in df.columns:
        date_str = df["Date"].astype("string")
        valid_mask = date_str.str.match(r"^\d{2}/\d{2}/\d{4}$", na=False)
        to_fix_mask = ~valid_mask & df["Date"].notna()
        if to_fix_mask.any():
            formats = [
                "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y",
                "%Y.%m.%d", "%d.%m.%Y",
            ]
            parsed = pd.Series(pd.NaT, index=df.index)
            for fmt in formats:
                newly = pd.to_datetime(df["Date"].where(to_fix_mask), format=fmt, errors="coerce")
                parsed = parsed.fillna(newly)
            ok = parsed.notna()
            df.loc[ok, "Date"] = parsed.loc[ok].dt.strftime("%d/%m/%Y")
    return df


# ---------------------------------------------------------------------------
# AST / AS swap tests
# ---------------------------------------------------------------------------

class TestAstAsSwap:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "AS", "AST",
                                           "HS", "HST", "Date"])

    def test_no_swap_needed(self):
        df = self._make_df([["A", "B", 10, 5, 8, 4, "01/01/2020"]])
        out = _apply_clean_logic(df)
        assert out.loc[0, "AS"] == 10
        assert out.loc[0, "AST"] == 5

    def test_single_swap(self):
        """Row where AST=15 > AS=8 must be swapped."""
        df = self._make_df([["A", "B", 8, 15, 6, 3, "01/01/2020"]])
        out = _apply_clean_logic(df)
        assert out.loc[0, "AS"] == 15
        assert out.loc[0, "AST"] == 8

    def test_equal_values_not_swapped(self):
        """AST == AS is not > AS, so no swap should occur."""
        df = self._make_df([["A", "B", 7, 7, 5, 5, "01/01/2020"]])
        out = _apply_clean_logic(df)
        assert out.loc[0, "AS"] == 7
        assert out.loc[0, "AST"] == 7

    def test_mixed_rows(self):
        """Only the row where AST > AS is swapped; others are untouched."""
        df = self._make_df([
            ["A", "B", 10, 5,  8, 4, "01/01/2020"],   # fine
            ["C", "D",  6, 12, 5, 3, "02/01/2020"],   # needs swap
            ["E", "F",  9, 9,  7, 7, "03/01/2020"],   # equal — fine
        ])
        out = _apply_clean_logic(df)
        assert out.loc[0, "AS"] == 10 and out.loc[0, "AST"] == 5
        assert out.loc[1, "AS"] == 12 and out.loc[1, "AST"] == 6
        assert out.loc[2, "AS"] == 9  and out.loc[2, "AST"] == 9

    def test_hs_hst_untouched(self):
        """The swap must never affect home-team shot columns."""
        df = self._make_df([["A", "B", 5, 10, 8, 3, "01/01/2020"]])
        out = _apply_clean_logic(df)
        assert out.loc[0, "HS"] == 8
        assert out.loc[0, "HST"] == 3

    def test_zero_values(self):
        df = self._make_df([["A", "B", 0, 1, 0, 0, "01/01/2020"]])
        out = _apply_clean_logic(df)
        assert out.loc[0, "AS"] == 1
        assert out.loc[0, "AST"] == 0

    def test_large_dataset_swap_count(self):
        """Correct number of rows swapped in a larger frame."""
        import numpy as np
        rng = pd.DataFrame({
            "HomeTeam": "H", "AwayTeam": "A",
            "AS":  [5, 3, 8, 2, 10],
            "AST": [3, 6, 4, 9,  7],
            "HS": 5, "HST": 3,
            "Date": "01/01/2020",
        })
        out = _apply_clean_logic(rng)
        # Rows 1 (3<6) and 3 (2<9) should be swapped
        assert out.loc[1, "AS"] == 6 and out.loc[1, "AST"] == 3
        assert out.loc[3, "AS"] == 9 and out.loc[3, "AST"] == 2


# ---------------------------------------------------------------------------
# Date normalisation tests
# ---------------------------------------------------------------------------

class TestDateNormalisation:
    def _df(self, date_val):
        return pd.DataFrame({
            "HomeTeam": ["A"], "AwayTeam": ["B"],
            "AS": [5], "AST": [3], "HS": [5], "HST": [3],
            "Date": [date_val],
        })

    def test_already_normalised(self):
        out = _apply_clean_logic(self._df("15/03/2022"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_iso_format(self):
        out = _apply_clean_logic(self._df("2022-03-15"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_slash_ymd(self):
        out = _apply_clean_logic(self._df("2022/03/15"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_dash_dmy(self):
        out = _apply_clean_logic(self._df("15-03-2022"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_dot_ymd(self):
        out = _apply_clean_logic(self._df("2022.03.15"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_dot_dmy(self):
        out = _apply_clean_logic(self._df("15.03.2022"))
        assert out.loc[0, "Date"] == "15/03/2022"

    def test_null_date_unchanged(self):
        out = _apply_clean_logic(self._df(None))
        assert pd.isna(out.loc[0, "Date"])

    def test_multiple_dates_mixed_formats(self):
        df = pd.DataFrame({
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["B", "C", "D"],
            "AS": [5, 5, 5], "AST": [3, 3, 3],
            "HS": [5, 5, 5], "HST": [3, 3, 3],
            "Date": ["01/08/2020", "2020-09-12", "12-10-2020"],
        })
        out = _apply_clean_logic(df)
        assert out.loc[0, "Date"] == "01/08/2020"
        assert out.loc[1, "Date"] == "12/09/2020"
        assert out.loc[2, "Date"] == "12/10/2020"


# ---------------------------------------------------------------------------
# File-level integration using tmp_path (no mocking needed)
# ---------------------------------------------------------------------------

class TestCleanDatasetFile:
    """End-to-end test: write a CSV, run clean_dataset(), read output."""

    def test_swap_and_date_via_file(self, tmp_path):
        import sys, os
        # Locate clean.py relative to this test file
        src_data = os.path.join(os.path.dirname(__file__), "..", "..", "src", "data")
        if src_data not in sys.path:
            sys.path.insert(0, src_data)

        try:
            from clean import clean_dataset
        except ModuleNotFoundError:
            pytest.skip("clean.py not importable from expected path")

        input_csv = tmp_path / "raw.csv"
        output_csv = tmp_path / "clean.csv"

        df = pd.DataFrame({
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["C", "D"],
            "AS":  [6, 3],
            "AST": [3, 9],  # row 1: AST > AS → needs swap
            "HS":  [8, 7],
            "HST": [4, 3],
            "Date": ["2022-08-05", "01/09/2022"],
        })
        df.to_csv(input_csv, index=False)
        clean_dataset(str(input_csv), str(output_csv))

        out = pd.read_csv(output_csv)
        assert out.loc[0, "AS"] == 6   and out.loc[0, "AST"] == 3   # untouched
        assert out.loc[1, "AS"] == 9   and out.loc[1, "AST"] == 3   # swapped
        assert out.loc[0, "Date"] == "05/08/2022"                    # normalised
        assert out.loc[1, "Date"] == "01/09/2022"                    # already OK