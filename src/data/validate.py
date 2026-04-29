import argparse
import os
import sys
from io import StringIO

import pandas as pd

# ── Output capture ──────────────────────────────────────────────────────────

class Tee:
    """Write output to both console and a string buffer."""
    def __init__(self):
        self.buffer = StringIO()
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self.buffer.write(msg)

    def flush(self):
        self._stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str):
    print(f"\n── {title} {'─' * (60 - len(title))}")


def print_issue_rows(df: pd.DataFrame, mask: pd.Series, columns=None, max_rows: int = 10):
    issue_rows = df[mask]
    if issue_rows.empty:
        print("  ✓ No issue rows found.")
        return

    print(f"  Issue rows ({len(issue_rows)} total):")
    if columns is not None:
        print(issue_rows[columns].head(max_rows).to_string(index=False))
    else:
        print(issue_rows.head(max_rows).to_string(index=False))
    if len(issue_rows) > max_rows:
        print(f"  ... showing first {max_rows} row(s)")


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset."""
    print(f"[INFO] Loading single file: {filepath}")
    df = pd.read_csv(filepath)
    df["_source_file"] = os.path.basename(filepath)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse mixed date formats robustly."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")

    n_bad = df["Date"].isna().sum()
    if n_bad:
        raise ValueError(f"Unparseable Date values found in {n_bad} row(s).")

    print("[INFO] Date parsed successfully with mixed format parsing.")
    return df


# ── Validation checks ────────────────────────────────────────────────────────

def check_shape(df: pd.DataFrame):
    section("1. SHAPE & BASIC COUNTS")
    rows, cols = df.shape
    print(f"  Total rows    : {rows:,}")
    print(f"  Total columns : {cols}")
    meets_rows = "✓ PASS" if rows >= 3000 else "✗ FAIL – need ≥ 3,000 rows"
    meets_cols = "✓ PASS" if cols >= 7 else "✗ FAIL – need ≥ 7 features"
    print(f"\n  Row requirement (≥3,000)    : {meets_rows}")
    print(f"  Column requirement (≥7)     : {meets_cols}")
    return rows, cols


def check_dtypes(df: pd.DataFrame):
    section("2. DATA TYPES")
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Dtype"]
    print(dtype_df.to_string(index=False))


def check_missing(df: pd.DataFrame):
    section("3. MISSING VALUES")
    missing = df.isnull().sum()
    pct = (df.isnull().mean() * 100).round(2)
    result = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    result = result[result["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    if result.empty:
        print("  ✓ No missing values detected.")
    else:
        print(result.to_string())
        print(f"\n  ⚠ {len(result)} column(s) have missing values.")
        print("\n  Rows containing missing values:")
        print_issue_rows(df, df.isnull().any(axis=1))
    return result


def check_duplicates(df: pd.DataFrame):
    section("4. DUPLICATE ROWS")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        print("  ✓ No duplicate rows found.")
    else:
        print(f"  ⚠ {n_dup} duplicate row(s) detected.")
        print("  Duplicate rows:")
        print(df[df.duplicated(keep=False)].to_string(index=False))
    return n_dup


def check_target(df: pd.DataFrame):
    section("5. TARGET VARIABLE: FTR (Full Time Result)")

    if "FTR" not in df.columns:
        print("  ✗ ERROR: 'FTR' column not found!")
        return

    subsection("Value counts")
    vc = df["FTR"].value_counts()
    pct = df["FTR"].value_counts(normalize=True).mul(100).round(2)
    result = pd.DataFrame({"Count": vc, "Percentage": pct})
    print(result.to_string())

    unexpected = set(df["FTR"].dropna().unique()) - {"H", "D", "A"}
    if unexpected:
        print(f"\n  ⚠ Unexpected FTR values: {unexpected}")
        print("  Rows with unexpected FTR values:")
        print_issue_rows(df, ~df["FTR"].isin(["H", "D", "A"]), columns=["Date", "HomeTeam", "AwayTeam", "FTR"])
    else:
        print("\n  ✓ All FTR values are valid (H / D / A).")

    subsection("Class imbalance assessment")
    majority = pct.max()
    minority = pct.min()
    ratio = majority / minority
    print(f"  Majority class : {pct.idxmax()} ({majority:.1f}%)")
    print(f"  Minority class : {pct.idxmin()} ({minority:.1f}%)")
    print(f"  Imbalance ratio: {ratio:.2f}:1")
    if ratio > 2:
        print("  ⚠ Notable imbalance – consider SMOTE or class-weight adjustment during training.")
    else:
        print("  ✓ Classes are reasonably balanced.")


def check_categorical_consistency(df: pd.DataFrame):
    section("6. CATEGORICAL CONSISTENCY")

    for col in ["FTR"]:
        if col in df.columns:
            vals = df[col].dropna().unique()
            print(f"  {col} unique values : {sorted(vals)}")

    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique()
        all_teams = sorted(list(all_teams))
        print(f"\n  Unique teams across all seasons ({len(all_teams)}):")
        for i, t in enumerate(all_teams):
            end = "\n" if (i + 1) % 5 == 0 else ""
            print(f"    {t:<25}", end=end)
        print()


def check_date_coverage(df: pd.DataFrame):
    section("7. DATE & SEASON COVERAGE")
    if "Date" not in df.columns:
        print("  ✗ 'Date' column not found.")
        return

    print(f"  Earliest match : {df['Date'].min().date()}")
    print(f"  Latest match   : {df['Date'].max().date()}")

    # Derive season from August cut-off
    df = df.copy()
    df["season_year"] = df["Date"].apply(
        lambda d: d.year if d.month >= 8 else d.year - 1
    )
    df["season_label"] = df["season_year"].apply(
        lambda y: f"{y}/{str(y+1)[-2:]}"
    )
    season_counts = df.groupby("season_label").size().reset_index(name="Matches")
    print(f"\n  Matches per season:")
    print(season_counts.to_string(index=False))

    expected = 380
    off = season_counts[season_counts["Matches"] != expected]
    if off.empty:
        print(f"\n  ✓ Every season has exactly {expected} matches.")
    else:
        print(f"\n  ⚠ Seasons with unexpected match counts (expected {expected}):")
        print(off.to_string(index=False))


def check_numeric_ranges(df: pd.DataFrame):
    section("8. NUMERIC RANGE SANITY CHECKS")

    numeric_checks = {
        "FTHG": (0, 20, "Full-time home goals"),
        "FTAG": (0, 20, "Full-time away goals"),
        "HS":   (0, 50, "Home shots"),
        "AS":   (0, 50, "Away shots"),
        "HST":  (0, 30, "Home shots on target"),
        "AST":  (0, 30, "Away shots on target"),
        "HF":   (0, 40, "Home fouls"),
        "AF":   (0, 40, "Away fouls"),
        "HC":   (0, 20, "Home corners"),
        "AC":   (0, 20, "Away corners"),
        "HY":   (0, 11, "Home yellow cards"),
        "AY":   (0, 11, "Away yellow cards"),
        "HR":   (0, 3,  "Home red cards"),
        "AR":   (0, 3,  "Away red cards"),
    }

    all_ok = True
    for col, (lo, hi, label) in numeric_checks.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        violations = series[(series < lo) | (series > hi)]
        status = "✓" if violations.empty else f"⚠ {len(violations)} violation(s)"
        min_v, max_v = series.min(), series.max()
        print(f"  {col:<6} ({label:<30}): min={min_v:>4.0f}  max={max_v:>4.0f}  {status}")
        if not violations.empty:
            all_ok = False
            print(f"    Rows with invalid {col} values:")
            print_issue_rows(df, (df[col] < lo) | (df[col] > hi), columns=[c for c in ["Date", "HomeTeam", "AwayTeam", col] if c in df.columns])

    if all_ok:
        print("\n  ✓ All numeric columns within expected ranges.")

    subsection("Logical consistency: HST ≤ HS and AST ≤ AS")
    if all(c in df.columns for c in ["HS", "HST", "AS", "AST"]):
        bad_hst = df[df["HST"] > df["HS"]]
        bad_ast = df[df["AST"] > df["AS"]]
        print(f"  Rows where HST > HS : {len(bad_hst)}")
        print(f"  Rows where AST > AS : {len(bad_ast)}")
        if bad_hst.empty and bad_ast.empty:
            print("  ✓ Shots-on-target ≤ total shots in all rows.")
        else:
            if not bad_hst.empty:
                print("  Rows where HST > HS:")
                print(bad_hst[[c for c in ["Date", "HomeTeam", "AwayTeam", "HS", "HST"] if c in bad_hst.columns]].to_string(index=False))
            if not bad_ast.empty:
                print("  Rows where AST > AS:")
                print(bad_ast[[c for c in ["Date", "HomeTeam", "AwayTeam", "AS", "AST"] if c in bad_ast.columns]].to_string(index=False))


def check_descriptive_stats(df: pd.DataFrame):
    section("9. DESCRIPTIVE STATISTICS (numeric columns)")
    numeric_cols = [c for c in [
        "FTHG", "FTAG",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR"
    ] if c in df.columns]

    print(df[numeric_cols].describe().round(2).to_string())


def check_referee(df: pd.DataFrame):
    section("10. REFEREE OVERVIEW")
    if "Referee" not in df.columns:
        print("  'Referee' column not found.")
        return
    n_refs = df["Referee"].nunique()
    print(f"  Unique referees : {n_refs}")
    top10 = df["Referee"].value_counts().head(10)
    print("\n  Top 10 referees by matches officiated:")
    print(top10.to_string())


def check_feature_completeness(df: pd.DataFrame):
    section("11. COLUMN COMPLETENESS SUMMARY")
    expected_cols = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "Referee",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR",
    ]
    print(f"  {'Column':<10} {'Present':>10}")
    print("  " + "-" * 22)
    for col in expected_cols:
        present = "✓" if col in df.columns else "✗ MISSING"
        print(f"  {col:<10} {present:>10}")

    extra_cols = [c for c in df.columns if c not in expected_cols + ["_source_file"]]
    if extra_cols:
        print(f"\n  Additional columns found: {extra_cols}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Premier League dataset validation script (CMPS344 Phase 2)"
    )
    parser.add_argument("--input", default="./dataset.csv", help="Path to the input CSV file")
    parser.add_argument("--output", default="./validation_report.txt", help="Path to save the validation report")
    args = parser.parse_args()

    tee = Tee()
    sys.stdout = tee

    print("=" * 70)
    print("  PREMIER LEAGUE DATASET VALIDATION REPORT")
    print("  CMPS344 Applied Data Science – Phase 2")
    print("=" * 70)

    # Load
    df = load_data(args.input)
    df = parse_dates(df)

    # Run all checks
    rows, cols = check_shape(df)
    check_dtypes(df)
    check_missing(df)
    check_duplicates(df)
    check_target(df)
    check_categorical_consistency(df)
    check_date_coverage(df)
    check_numeric_ranges(df)
    check_descriptive_stats(df)
    check_referee(df)
    check_feature_completeness(df)

    section("VALIDATION COMPLETE")
    print(f"  Report saved to {args.output}")

    sys.stdout = tee._stdout
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())

    print(f"\n[INFO] Validation complete. Results written to {args.output}")


if __name__ == "__main__":
    main()
