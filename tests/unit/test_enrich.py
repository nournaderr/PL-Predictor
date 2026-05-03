"""
tests/unit/test_enrich.py
Unit tests for src/features/enrich.py
"""
import sys, os
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make enrich importable regardless of working directory
# ---------------------------------------------------------------------------
_FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "features")
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

try:
    from enrich import (
        safe_div,
        avg,
        TeamStats,
        record_match,
        build_table,
        split_into_seasons,
        SEASON_SIZE,
        SEASON_COUNT,
    )
    _IMPORT_OK = True
except ModuleNotFoundError:
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="enrich.py not importable")


# ---------------------------------------------------------------------------
# safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert safe_div(9.0, 3) == pytest.approx(3.0)

    def test_zero_denominator_returns_zero(self):
        assert safe_div(100.0, 0) == 0.0

    def test_zero_numerator(self):
        assert safe_div(0.0, 5) == 0.0

    def test_fractional_result(self):
        assert safe_div(1.0, 3) == pytest.approx(1 / 3)

    def test_large_values(self):
        assert safe_div(1_000_000.0, 1_000) == pytest.approx(1_000.0)

    def test_negative_numerator(self):
        assert safe_div(-6.0, 3) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# avg
# ---------------------------------------------------------------------------

class TestAvg:
    def test_simple_average(self):
        assert avg([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_empty_returns_zero(self):
        assert avg([]) == 0.0

    def test_single_value(self):
        assert avg([7]) == pytest.approx(7.0)

    def test_all_zeros(self):
        assert avg([0, 0, 0]) == pytest.approx(0.0)

    def test_floats(self):
        assert avg([1.5, 2.5, 3.0]) == pytest.approx(7.0 / 3)

    def test_generator_input(self):
        assert avg(x for x in [2, 4, 6]) == pytest.approx(4.0)

    def test_deque_input(self):
        from collections import deque
        d = deque([10, 20, 30], maxlen=5)
        assert avg(d) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TeamStats + record_match
# ---------------------------------------------------------------------------

class TestRecordMatch:
    def _fresh(self) -> "TeamStats":
        return TeamStats()

    def test_home_win_increments_points(self):
        s = self._fresh()
        record_match(s, is_home=True, points=3, goals_for=2, goals_against=0,
                     shots=10, shots_on_target=5, fouls=8, corners=4, yellow=1, red=0)
        assert s.points_total == 3
        assert s.home_points == 3
        assert s.away_points == 0

    def test_clean_sheet_recorded(self):
        s = self._fresh()
        record_match(s, is_home=True, points=3, goals_for=1, goals_against=0,
                     shots=8, shots_on_target=3, fouls=6, corners=3, yellow=0, red=0)
        assert s.clean_sheets == 1
        assert s.home_clean_sheets == 1

    def test_no_clean_sheet_when_goals_conceded(self):
        s = self._fresh()
        record_match(s, is_home=False, points=0, goals_for=0, goals_against=2,
                     shots=5, shots_on_target=2, fouls=5, corners=2, yellow=1, red=0)
        assert s.clean_sheets == 0
        assert s.away_clean_sheets == 0

    def test_away_match_increments_away_stats(self):
        s = self._fresh()
        record_match(s, is_home=False, points=1, goals_for=1, goals_against=1,
                     shots=7, shots_on_target=3, fouls=9, corners=5, yellow=2, red=0)
        assert s.away_matches == 1
        assert s.away_points == 1
        assert s.home_matches == 0

    def test_last5_deque_capped_at_5(self):
        s = self._fresh()
        for i in range(7):
            record_match(s, is_home=True, points=3, goals_for=2, goals_against=0,
                         shots=10, shots_on_target=5, fouls=5, corners=3, yellow=1, red=0)
        assert len(s.last5_points) == 5
        assert len(s.last5_goals_for) == 5

    def test_matches_total_increments(self):
        s = self._fresh()
        for _ in range(3):
            record_match(s, is_home=True, points=1, goals_for=1, goals_against=1,
                         shots=5, shots_on_target=2, fouls=4, corners=2, yellow=1, red=0)
        assert s.matches_total == 3

    def test_shot_and_card_deques_populated(self):
        s = self._fresh()
        record_match(s, is_home=True, points=3, goals_for=2, goals_against=0,
                     shots=12, shots_on_target=6, fouls=7, corners=4, yellow=2, red=1)
        assert list(s.last5_shots) == [12]
        assert list(s.last5_shots_on_target) == [6]
        assert list(s.last5_fouls) == [7]
        assert list(s.last5_corners) == [4]
        assert list(s.last5_yellow) == [2]
        assert list(s.last5_red) == [1]

    def test_draw_points(self):
        s = self._fresh()
        record_match(s, is_home=True, points=1, goals_for=1, goals_against=1,
                     shots=6, shots_on_target=3, fouls=6, corners=3, yellow=1, red=0)
        assert s.points_total == 1

    def test_loss_zero_points(self):
        s = self._fresh()
        record_match(s, is_home=True, points=0, goals_for=0, goals_against=3,
                     shots=4, shots_on_target=1, fouls=8, corners=2, yellow=3, red=0)
        assert s.points_total == 0
        assert s.clean_sheets == 0


# ---------------------------------------------------------------------------
# build_table
# ---------------------------------------------------------------------------

class TestBuildTable:
    def _make_stats(self, pts, gf, ga):
        s = TeamStats()
        s.points_total = pts
        s.goals_for = gf
        s.goals_against = ga
        s.matches_total = 5
        return s

    def test_positions_ordered_by_points(self):
        team_stats = {
            "Alpha": self._make_stats(12, 10, 5),
            "Beta":  self._make_stats(9,  8,  6),
            "Gamma": self._make_stats(6,  5,  7),
        }
        pos, _, _ = build_table(team_stats)
        assert pos["Alpha"] == 1
        assert pos["Beta"]  == 2
        assert pos["Gamma"] == 3

    def test_goal_diff_tiebreaker(self):
        """Two teams with same points; better GD ranks higher."""
        team_stats = {
            "Team1": self._make_stats(9, 10, 4),   # GD = +6
            "Team2": self._make_stats(9,  7, 5),   # GD = +2
        }
        pos, _, _ = build_table(team_stats)
        assert pos["Team1"] == 1
        assert pos["Team2"] == 2

    def test_leader_has_zero_points_above(self):
        team_stats = {
            "Leader":  self._make_stats(15, 12, 4),
            "Second":  self._make_stats(10, 8,  6),
        }
        _, points_above, _ = build_table(team_stats)
        assert points_above["Leader"] == 0.0

    def test_last_place_has_zero_points_below(self):
        team_stats = {
            "First": self._make_stats(15, 12, 4),
            "Last":  self._make_stats(3,  4, 15),
        }
        _, _, points_below = build_table(team_stats)
        assert points_below["Last"] == 0.0

    def test_points_gap_correct(self):
        team_stats = {
            "A": self._make_stats(20, 15, 5),
            "B": self._make_stats(14, 10, 8),
            "C": self._make_stats(8,   6, 12),
        }
        pos, above, below = build_table(team_stats)
        # B is 6 pts behind A, 6 pts ahead of C
        assert above["B"] == 6.0
        assert below["B"] == 6.0

    def test_single_team(self):
        team_stats = {"Solo": self._make_stats(10, 8, 4)}
        pos, above, below = build_table(team_stats)
        assert pos["Solo"] == 1
        assert above["Solo"] == 0.0
        assert below["Solo"] == 0.0

    def test_all_teams_get_positions(self):
        team_stats = {f"Team{i}": self._make_stats(i * 3, i * 2, 5) for i in range(1, 6)}
        pos, _, _ = build_table(team_stats)
        assert set(pos.keys()) == set(team_stats.keys())
        assert sorted(pos.values()) == list(range(1, 6))


# ---------------------------------------------------------------------------
# split_into_seasons
# ---------------------------------------------------------------------------

class TestSplitIntoSeasons:
    def _make_df(self, n_rows):
        return pd.DataFrame({
            "HomeTeam": ["A"] * n_rows,
            "AwayTeam": ["B"] * n_rows,
            "Date": ["01/01/2020"] * n_rows,
        })

    def test_correct_number_of_seasons(self):
        df = self._make_df(SEASON_SIZE * SEASON_COUNT)
        seasons = split_into_seasons(df)
        assert len(seasons) == SEASON_COUNT

    def test_each_season_has_correct_size(self):
        df = self._make_df(SEASON_SIZE * SEASON_COUNT)
        seasons = split_into_seasons(df)
        for s in seasons:
            assert len(s) == SEASON_SIZE

    def test_raises_on_too_few_rows(self):
        df = self._make_df(SEASON_SIZE * SEASON_COUNT - 1)
        with pytest.raises(ValueError, match="Expected at least"):
            split_into_seasons(df)

    def test_extra_rows_truncated(self, capsys):
        extra = 10
        df = self._make_df(SEASON_SIZE * SEASON_COUNT + extra)
        seasons = split_into_seasons(df)
        assert len(seasons) == SEASON_COUNT
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_seasons_are_non_overlapping(self):
        df = pd.DataFrame({
            "HomeTeam": ["A"] * (SEASON_SIZE * SEASON_COUNT),
            "AwayTeam": ["B"] * (SEASON_SIZE * SEASON_COUNT),
            "Date": ["01/01/2020"] * (SEASON_SIZE * SEASON_COUNT),
            "id": range(SEASON_SIZE * SEASON_COUNT),
        })
        seasons = split_into_seasons(df)
        all_ids = []
        for s in seasons:
            all_ids.extend(s["id"].tolist())
        assert len(all_ids) == len(set(all_ids)), "Seasons must not share rows"