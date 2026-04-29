from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


INPUT_FILE_NAMES = [
    "2015-16.csv",
    "2016-17.csv",
    "2017-18.csv",
    "2018-19.csv",
    "2019-20.csv",
    "2020-21.csv",
    "2021-22.csv",
    "2022-23.csv",
    "2023-24.csv",
    "2024-25.csv",
]


BETTING_PREFIXES = (
    "B365",
    "BW",
    "IW",
    "PS",
    "WH",
    "VC",
    "BF",
    "BFE",
    "1XB",
    "P",
    "Max",
    "Avg",
)


HALFTIME_COLUMNS = {"HTHG", "HTAG", "HTR"}


def is_betting_column(column_name: str) -> bool:
    if column_name == "Div":
        return True

    if column_name.startswith(BETTING_PREFIXES):
        return True

    if any(token in column_name for token in (">", "<", "AH")):
        return True

    if column_name.endswith(("CH", "CD", "CA")):
        return True

    return False


def get_input_files(input: Path) -> List[Path]:
    files = [input / name for name in INPUT_FILE_NAMES]
    missing = [str(p.name) for p in files if not p.is_file()]
    if missing:
        missing_csv = ", ".join(missing)
        raise FileNotFoundError(f"Missing required input files: {missing_csv}")
    return files


def common_columns(csv_files: List[Path]) -> List[str]:
    first_cols = list(pd.read_csv(csv_files[0], nrows=0).columns)
    common = set(first_cols)

    for file_path in csv_files[1:]:
        cols = set(pd.read_csv(file_path, nrows=0).columns)
        common &= cols

    # Keep the original order from the first file.
    ordered_common = [c for c in first_cols if c in common]
    if not ordered_common:
        raise ValueError("No common columns exist across all CSV files.")
    return ordered_common


def combine_files(csv_files: List[Path], keep_cols: List[str]) -> pd.DataFrame:
    parts = [pd.read_csv(file_path, usecols=keep_cols) for file_path in csv_files]
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine all CSV files in a folder into one dataset using only columns "
            "that exist in every file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Folder containing CSV files (default: current folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset.csv"),
        help="Output CSV path (default: dataset.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input = args.input.resolve()
    output_path = args.output.resolve()

    csv_files = get_input_files(input)
    keep_cols = common_columns(csv_files)
    filtered_cols = [
        c for c in keep_cols
        if c not in HALFTIME_COLUMNS and not is_betting_column(c)
    ]
    if not filtered_cols:
        raise ValueError("No non-betting common columns remain after filtering.")
    combined = combine_files(csv_files, filtered_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Using {len(csv_files)} configured input files.")
    print(f"Keeping {len(filtered_cols)} non-betting common columns.")
    print(f"Removed {len(keep_cols) - len(filtered_cols)} columns.")
    print(f"Combined rows: {len(combined)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
