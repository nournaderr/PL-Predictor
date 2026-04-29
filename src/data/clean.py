import argparse

import pandas as pd


def clean_dataset(input_file: str, output_file: str) -> None:
    """
    Load dataset, find rows where AST > AS, and swap AS and AST values.
    Normalize Date values to DD/MM/YYYY when needed.
    Save the modified dataset to output_file.
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Find rows where AST > AS
    mask = df["AST"] > df["AS"]
    num_fixed = mask.sum()
    
    # Swap AS and AST values where the condition is true
    df.loc[mask, ["AS", "AST"]] = df.loc[mask, ["AST", "AS"]].values

    # Normalize dates that do not already match DD/MM/YYYY
    date_fixed = 0
    date_failed = 0
    if "Date" in df.columns:
        date_series = df["Date"]
        date_str = date_series.astype("string")
        valid_mask = date_str.str.match(r"^\d{2}/\d{2}/\d{4}$", na=False)
        to_fix_mask = ~valid_mask & date_series.notna()

        if to_fix_mask.any():
            to_parse = date_series.where(to_fix_mask)
            parsed = pd.Series(pd.NaT, index=date_series.index)
            formats = [
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%m/%d/%Y",
                "%m-%d-%Y",
                "%d/%m/%y",
                "%m/%d/%y",
                "%Y.%m.%d",
                "%d.%m.%Y",
            ]
            for fmt in formats:
                newly_parsed = pd.to_datetime(
                    to_parse,
                    format=fmt,
                    errors="coerce",
                )
                parsed = parsed.fillna(newly_parsed)
            parsed_mask = parsed.notna()
            df.loc[parsed_mask, "Date"] = parsed.loc[parsed_mask].dt.strftime(
                "%d/%m/%Y"
            )
            date_fixed = int(parsed_mask.sum())
            date_failed = int(to_fix_mask.sum() - date_fixed)
    
    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    
    print(f"Read: {input_file}")
    print(f"Fixed {num_fixed} rows where AST > AS")
    print(f"Normalized {date_fixed} dates to DD/MM/YYYY")
    if date_failed:
        print(f"Warning: {date_failed} dates could not be parsed")
    print(f"Saved: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean dataset by fixing rows where AST > AS."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(args.input, args.output)
