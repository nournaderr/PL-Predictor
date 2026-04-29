import argparse
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Tuple

import pandas as pd


SEASON_SIZE = 380
SEASON_COUNT = 10


def safe_div(numerator: float, denominator: int) -> float:
	return float(numerator) / denominator if denominator else 0.0


def avg(values: Iterable[float]) -> float:
	values_list = list(values)
	return float(sum(values_list) / len(values_list)) if values_list else 0.0


@dataclass
class TeamStats:
	points_total: int = 0
	matches_total: int = 0
	goals_for: int = 0
	goals_against: int = 0
	clean_sheets: int = 0

	home_points: int = 0
	home_matches: int = 0
	home_goals_for: int = 0
	home_goals_against: int = 0
	home_clean_sheets: int = 0

	away_points: int = 0
	away_matches: int = 0
	away_goals_for: int = 0
	away_goals_against: int = 0
	away_clean_sheets: int = 0

	last5_points: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_points_home: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_points_away: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

	last5_goals_for: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_goals_against: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_goals_for_home: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_goals_against_home: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_goals_for_away: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_goals_against_away: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

	last5_clean_sheets: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_clean_sheets_home: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_clean_sheets_away: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

	last5_shots: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_shots_on_target: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_fouls: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_corners: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_yellow: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
	last5_red: Deque[int] = field(default_factory=lambda: deque(maxlen=5))


def record_match(
	stats: TeamStats,
	*,
	is_home: bool,
	points: int,
	goals_for: int,
	goals_against: int,
	shots: int,
	shots_on_target: int,
	fouls: int,
	corners: int,
	yellow: int,
	red: int,
) -> None:
	stats.matches_total += 1
	stats.points_total += points
	stats.goals_for += goals_for
	stats.goals_against += goals_against
	if goals_against == 0:
		stats.clean_sheets += 1

	stats.last5_points.append(points)
	stats.last5_goals_for.append(goals_for)
	stats.last5_goals_against.append(goals_against)
	stats.last5_clean_sheets.append(1 if goals_against == 0 else 0)

	stats.last5_shots.append(shots)
	stats.last5_shots_on_target.append(shots_on_target)
	stats.last5_fouls.append(fouls)
	stats.last5_corners.append(corners)
	stats.last5_yellow.append(yellow)
	stats.last5_red.append(red)

	if is_home:
		stats.home_matches += 1
		stats.home_points += points
		stats.home_goals_for += goals_for
		stats.home_goals_against += goals_against
		if goals_against == 0:
			stats.home_clean_sheets += 1

		stats.last5_points_home.append(points)
		stats.last5_goals_for_home.append(goals_for)
		stats.last5_goals_against_home.append(goals_against)
		stats.last5_clean_sheets_home.append(1 if goals_against == 0 else 0)
	else:
		stats.away_matches += 1
		stats.away_points += points
		stats.away_goals_for += goals_for
		stats.away_goals_against += goals_against
		if goals_against == 0:
			stats.away_clean_sheets += 1

		stats.last5_points_away.append(points)
		stats.last5_goals_for_away.append(goals_for)
		stats.last5_goals_against_away.append(goals_against)
		stats.last5_clean_sheets_away.append(1 if goals_against == 0 else 0)


def build_table(team_stats: Dict[str, TeamStats]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
	table: List[Tuple[str, int, int, int]] = []
	for team, stats in team_stats.items():
		goal_diff = stats.goals_for - stats.goals_against
		table.append((team, stats.points_total, goal_diff, stats.goals_for))

	table.sort(key=lambda row: (-row[1], -row[2], -row[3], row[0]))

	positions: Dict[str, int] = {}
	points_above: Dict[str, float] = {}
	points_below: Dict[str, float] = {}

	for index, (team, points, _gd, _gf) in enumerate(table):
		positions[team] = index + 1

		if index == 0:
			points_above[team] = 0.0
		else:
			points_above[team] = float(table[index - 1][1] - points)

		if index == len(table) - 1:
			points_below[team] = 0.0
		else:
			points_below[team] = float(points - table[index + 1][1])

	return positions, points_above, points_below


def build_feature_row(
	row: pd.Series,
	home_stats: TeamStats,
	away_stats: TeamStats,
	positions: Dict[str, int],
	points_above: Dict[str, float],
	points_below: Dict[str, float],
) -> Dict[str, float]:
	home_team = row["HomeTeam"]
	away_team = row["AwayTeam"]

	return {
		"Date": row["Date"],
		"HomeTeam": home_team,
		"AwayTeam": away_team,
		"Referee": row["Referee"],
		"HPPG": safe_div(home_stats.points_total, home_stats.matches_total),
		"HPPGH": safe_div(home_stats.home_points, home_stats.home_matches),
		"HPPG_FORM": avg(home_stats.last5_points),
		"HPPGH_FORM": avg(home_stats.last5_points_home),
		"APPG": safe_div(away_stats.points_total, away_stats.matches_total),
		"APPGA": safe_div(away_stats.away_points, away_stats.away_matches),
		"APPG_FORM": avg(away_stats.last5_points),
		"APPGA_FORM": avg(away_stats.last5_points_away),
		"HGS": safe_div(home_stats.goals_for, home_stats.matches_total),
		"HGSH": safe_div(home_stats.home_goals_for, home_stats.home_matches),
		"HGS_FORM": avg(home_stats.last5_goals_for),
		"HGSH_FORM": avg(home_stats.last5_goals_for_home),
		"AGS": safe_div(away_stats.goals_for, away_stats.matches_total),
		"AGSA": safe_div(away_stats.away_goals_for, away_stats.away_matches),
		"AGS_FORM": avg(away_stats.last5_goals_for),
		"AGSA_FORM": avg(away_stats.last5_goals_for_away),
		"HGC": safe_div(home_stats.goals_against, home_stats.matches_total),
		"HGCH": safe_div(home_stats.home_goals_against, home_stats.home_matches),
		"HGC_FORM": avg(home_stats.last5_goals_against),
		"HGCH_FORM": avg(home_stats.last5_goals_against_home),
		"AGC": safe_div(away_stats.goals_against, away_stats.matches_total),
		"AGCA": safe_div(away_stats.away_goals_against, away_stats.away_matches),
		"AGC_FORM": avg(away_stats.last5_goals_against),
		"AGCA_FORM": avg(away_stats.last5_goals_against_away),
		"HCS": safe_div(home_stats.clean_sheets, home_stats.matches_total),
		"HCSH": safe_div(home_stats.home_clean_sheets, home_stats.home_matches),
		"HCS_FORM": avg(home_stats.last5_clean_sheets),
		"HCSH_FORM": avg(home_stats.last5_clean_sheets_home),
		"ACS": safe_div(away_stats.clean_sheets, away_stats.matches_total),
		"ACSA": safe_div(away_stats.away_clean_sheets, away_stats.away_matches),
		"ACS_FORM": avg(away_stats.last5_clean_sheets),
		"ACSA_FORM": avg(away_stats.last5_clean_sheets_away),
		"HPOS": positions[home_team],
		"APOS": positions[away_team],
		"HPDU": points_above[home_team],
		"HPDD": points_below[home_team],
		"APDU": points_above[away_team],
		"APDD": points_below[away_team],
		"HS_FORM": avg(home_stats.last5_shots),
		"AS_FORM": avg(away_stats.last5_shots),
		"HST_FORM": avg(home_stats.last5_shots_on_target),
		"AST_FORM": avg(away_stats.last5_shots_on_target),
		"HF_FORM": avg(home_stats.last5_fouls),
		"AF_FORM": avg(away_stats.last5_fouls),
		"HC_FORM": avg(home_stats.last5_corners),
		"AC_FORM": avg(away_stats.last5_corners),
		"HY_FORM": avg(home_stats.last5_yellow),
		"AY_FORM": avg(away_stats.last5_yellow),
		"HR_FORM": avg(home_stats.last5_red),
		"AR_FORM": avg(away_stats.last5_red),
		"FTR": row["FTR"],
	}


def update_stats_for_match(team_stats: Dict[str, TeamStats], row: pd.Series) -> None:
	home_team = row["HomeTeam"]
	away_team = row["AwayTeam"]
	home_goals = int(row["FTHG"])
	away_goals = int(row["FTAG"])
	result = row["FTR"]

	if result == "H":
		home_points, away_points = 3, 0
	elif result == "A":
		home_points, away_points = 0, 3
	else:
		home_points, away_points = 1, 1

	record_match(
		team_stats[home_team],
		is_home=True,
		points=home_points,
		goals_for=home_goals,
		goals_against=away_goals,
		shots=int(row["HS"]),
		shots_on_target=int(row["HST"]),
		fouls=int(row["HF"]),
		corners=int(row["HC"]),
		yellow=int(row["HY"]),
		red=int(row["HR"]),
	)

	record_match(
		team_stats[away_team],
		is_home=False,
		points=away_points,
		goals_for=away_goals,
		goals_against=home_goals,
		shots=int(row["AS"]),
		shots_on_target=int(row["AST"]),
		fouls=int(row["AF"]),
		corners=int(row["AC"]),
		yellow=int(row["AY"]),
		red=int(row["AR"]),
	)


def process_season(season_df: pd.DataFrame) -> List[Dict[str, float]]:
	teams = sorted(set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"])))
	team_stats = {team: TeamStats() for team in teams}

	output_rows: List[Dict[str, float]] = []

	for _date, day_df in season_df.groupby("Date", sort=False):
		positions, points_above, points_below = build_table(team_stats)

		for _, row in day_df.iterrows():
			home_stats = team_stats[row["HomeTeam"]]
			away_stats = team_stats[row["AwayTeam"]]
			output_rows.append(
				build_feature_row(row, home_stats, away_stats, positions, points_above, points_below)
			)

		for _, row in day_df.iterrows():
			update_stats_for_match(team_stats, row)

	return output_rows


def split_into_seasons(df: pd.DataFrame) -> List[pd.DataFrame]:
	expected_rows = SEASON_SIZE * SEASON_COUNT

	if len(df) < expected_rows:
		raise ValueError(
			f"Expected at least {expected_rows} rows for {SEASON_COUNT} seasons, got {len(df)}."
		)

	if len(df) > expected_rows:
		print(
			f"Warning: input has {len(df)} rows; only the first {expected_rows} rows will be processed.",
			file=sys.stderr,
		)
		df = df.iloc[:expected_rows].copy()

	return [
		df.iloc[i * SEASON_SIZE : (i + 1) * SEASON_SIZE].reset_index(drop=True)
		for i in range(SEASON_COUNT)
	]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
	seasons = split_into_seasons(df)

	all_rows: List[Dict[str, float]] = []
	for season_df in seasons:
		all_rows.extend(process_season(season_df))

	column_order = [
		"Date",
		"HomeTeam",
		"AwayTeam",
		"Referee",
		"HPPG",
		"HPPGH",
		"HPPG_FORM",
		"HPPGH_FORM",
		"APPG",
		"APPGA",
		"APPG_FORM",
		"APPGA_FORM",
		"HGS",
		"HGSH",
		"HGS_FORM",
		"HGSH_FORM",
		"AGS",
		"AGSA",
		"AGS_FORM",
		"AGSA_FORM",
		"HGC",
		"HGCH",
		"HGC_FORM",
		"HGCH_FORM",
		"AGC",
		"AGCA",
		"AGC_FORM",
		"AGCA_FORM",
		"HCS",
		"HCSH",
		"HCS_FORM",
		"HCSH_FORM",
		"ACS",
		"ACSA",
		"ACS_FORM",
		"ACSA_FORM",
		"HPOS",
		"APOS",
		"HPDU",
		"HPDD",
		"APDU",
		"APDD",
		"HS_FORM",
		"AS_FORM",
		"HST_FORM",
		"AST_FORM",
		"HF_FORM",
		"AF_FORM",
		"HC_FORM",
		"AC_FORM",
		"HY_FORM",
		"AY_FORM",
		"HR_FORM",
		"AR_FORM",
		"FTR",
	]

	return pd.DataFrame(all_rows, columns=column_order)


def validate_input(df: pd.DataFrame) -> None:
	required_columns = [
		"Date",
		"HomeTeam",
		"AwayTeam",
		"FTHG",
		"FTAG",
		"FTR",
		"Referee",
		"HS",
		"AS",
		"HST",
		"AST",
		"HF",
		"AF",
		"HC",
		"AC",
		"HY",
		"AY",
		"HR",
		"AR",
	]

	missing = [column for column in required_columns if column not in df.columns]
	if missing:
		raise ValueError(f"Input dataset is missing required columns: {', '.join(missing)}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate engineered match features for the Premier League dataset."
	)
	parser.add_argument("--input", required=True, help="Path to the cleaned input CSV.")
	parser.add_argument("--output", required=True, help="Path for the enriched output CSV.")
	args = parser.parse_args()

	df = pd.read_csv(args.input)
	validate_input(df)

	output_df = build_output(df)
	output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
	main()
