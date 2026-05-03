"""
tests/unit/test_construct.py
Unit tests for src/data/construct.py
"""
import sys, os
import pandas as pd
import pytest
from pathlib import Path

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

try:
    from construct import is_betting_column, common_columns, combine_files
    _IMPORT_OK = True
except ModuleNotFoundError:
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="construct.py not importable")


# ---------------------------------------------------------------------------
# is_betting_column
# ---------------------------------------------------------------------------

class TestIsBettingColumn:
    # -- should be identified as betting --
    def test_div_is_betting(self):
        assert is_betting_column("Div") is True

    def test_b365h_is_betting(self):
        assert is_betting_column("B365H") is True

    def test_b365d_is_betting(self):
        assert is_betting_column("B365D") is True

    def test_b365a_is_betting(self):
        assert is_betting_column("B365A") is True

    def test_bwh_is_betting(self):
        assert is_betting_column("BWH") is True

    def test_iwh_is_betting(self):
        assert is_betting_column("IWH") is True

    def test_psh_is_betting(self):
        assert is_betting_column("PSH") is True

    def test_whh_is_betting(self):
        assert is_betting_column("WHH") is True

    def test_vch_is_betting(self):
        assert is_betting_column("VCH") is True

    def test_maxh_is_betting(self):
        assert is_betting_column("MaxH") is True

    def test_avgh_is_betting(self):
        assert is_betting_column("AvgH") is True

    def test_greater_than_in_name_is_betting(self):
        assert is_betting_column("B365>2.5") is True

    def test_less_than_in_name_is_betting(self):
        assert is_betting_column("B365<2.5") is True

    def test_ah_in_name_is_betting(self):
        assert is_betting_column("B365AH") is True

    def test_ch_suffix_is_betting(self):
        assert is_betting_column("B365CH") is True

    def test_cd_suffix_is_betting(self):
        assert is_betting_column("B365CD") is True

    def test_ca_suffix_is_betting(self):
        assert is_betting_column("B365CA") is True

    # -- should NOT be identified as betting --
    def test_ftr_not_betting(self):
        assert is_betting_column("FTR") is False

    def test_fthg_not_betting(self):
        assert is_betting_column("FTHG") is False

    def test_ftag_not_betting(self):
        assert is_betting_column("FTAG") is False

    def test_hometeam_not_betting(self):
        assert is_betting_column("HomeTeam") is False

    def test_awayteam_not_betting(self):
        assert is_betting_column("AwayTeam") is False

    def test_referee_not_betting(self):
        assert is_betting_column("Referee") is False

    def test_hs_not_betting(self):
        assert is_betting_column("HS") is False

    def test_as_not_betting(self):
        assert is_betting_column("AS") is False

    def test_hst_not_betting(self):
        assert is_betting_column("HST") is False

    def test_ast_not_betting(self):
        assert is_betting_column("AST") is False

    def test_date_not_betting(self):
        assert is_betting_column("Date") is False

    def test_hy_not_betting(self):
        assert is_betting_column("HY") is False

    def test_ay_not_betting(self):
        assert is_betting_column("AY") is False

    def test_hr_not_betting(self):
        assert is_betting_column("HR") is False

    def test_ar_not_betting(self):
        assert is_betting_column("AR") is False

    def test_hf_not_betting(self):
        assert is_betting_column("HF") is False

    def test_hc_not_betting(self):
        assert is_betting_column("HC") is False


# ---------------------------------------------------------------------------
# common_columns (using tmp CSVs)
# ---------------------------------------------------------------------------

class TestCommonColumns:
    def _write_csv(self, path: Path, cols: list) -> Path:
        df = pd.DataFrame(columns=cols)
        df.to_csv(path, index=False)
        return path

    def test_all_same_columns(self, tmp_path):
        cols = ["Date", "HomeTeam", "AwayTeam", "FTR"]
        files = [self._write_csv(tmp_path / f"f{i}.csv", cols) for i in range(3)]
        result = common_columns(files)
        assert result == cols

    def test_intersection_only(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", ["Date", "HomeTeam", "AwayTeam", "FTR", "Extra"])
        f2 = self._write_csv(tmp_path / "b.csv", ["Date", "HomeTeam", "AwayTeam", "FTR"])
        result = common_columns([f1, f2])
        assert "Extra" not in result
        assert "FTR" in result

    def test_order_follows_first_file(self, tmp_path):
        cols = ["FTR", "Date", "HomeTeam", "AwayTeam"]
        files = [self._write_csv(tmp_path / f"f{i}.csv", cols) for i in range(2)]
        result = common_columns(files)
        assert result == cols

    def test_no_common_columns_raises(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", ["ColA"])
        f2 = self._write_csv(tmp_path / "b.csv", ["ColB"])
        with pytest.raises(ValueError):
            common_columns([f1, f2])


# ---------------------------------------------------------------------------
# combine_files
# ---------------------------------------------------------------------------

class TestCombineFiles:
    def _write_csv(self, path: Path, data: dict) -> Path:
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def test_row_count_is_sum(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", {"Date": ["d1", "d2"], "FTR": ["H", "D"]})
        f2 = self._write_csv(tmp_path / "b.csv", {"Date": ["d3"],       "FTR": ["A"]})
        result = combine_files([f1, f2], keep_cols=["Date", "FTR"])
        assert len(result) == 3

    def test_columns_match_keep_cols(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", {"Date": ["d1"], "FTR": ["H"], "X": [1]})
        f2 = self._write_csv(tmp_path / "b.csv", {"Date": ["d2"], "FTR": ["D"], "X": [2]})
        result = combine_files([f1, f2], keep_cols=["Date", "FTR"])
        assert list(result.columns) == ["Date", "FTR"]

    def test_index_is_reset(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", {"Date": ["d1", "d2"], "FTR": ["H", "H"]})
        f2 = self._write_csv(tmp_path / "b.csv", {"Date": ["d3"],       "FTR": ["A"]})
        result = combine_files([f1, f2], keep_cols=["Date", "FTR"])
        assert list(result.index) == list(range(len(result)))

    def test_single_file(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", {"Date": ["d1", "d2"], "FTR": ["H", "D"]})
        result = combine_files([f1], keep_cols=["Date", "FTR"])
        assert len(result) == 2

    def test_data_values_preserved(self, tmp_path):
        f1 = self._write_csv(tmp_path / "a.csv", {"Date": ["01/08/2020"], "FTR": ["H"]})
        f2 = self._write_csv(tmp_path / "b.csv", {"Date": ["02/08/2020"], "FTR": ["A"]})
        result = combine_files([f1, f2], keep_cols=["Date", "FTR"])
        assert result.iloc[0]["FTR"] == "H"
        assert result.iloc[1]["FTR"] == "A"