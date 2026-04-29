import argparse
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


NUMERICAL_COLS: List[str] = [
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

CATEGORICAL_COLS: List[str] = ["HomeTeam", "AwayTeam", "Referee", "FTR"]
TARGET_COL = "FTR"
TARGET_ENCODING: Dict[str, int] = {"H": 2, "D": 1, "A": 0}
PALETTE: Dict[str, str] = {"H": "#27ae60", "D": "#2980b9", "A": "#c0392b"}

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Points Per Game (Season)":  ["HPPG", "HPPGH", "APPG", "APPGA"],
    "Points Per Game (Form)":    ["HPPG_FORM", "HPPGH_FORM", "APPG_FORM", "APPGA_FORM"],
    "Goals Scored (Season)":     ["HGS", "HGSH", "AGS", "AGSA"],
    "Goals Scored (Form)":       ["HGS_FORM", "HGSH_FORM", "AGS_FORM", "AGSA_FORM"],
    "Goals Conceded (Season)":   ["HGC", "HGCH", "AGC", "AGCA"],
    "Goals Conceded (Form)":     ["HGC_FORM", "HGCH_FORM", "AGC_FORM", "AGCA_FORM"],
    "Clean Sheets (Season)":     ["HCS", "HCSH", "ACS", "ACSA"],
    "Clean Sheets (Form)":       ["HCS_FORM", "HCSH_FORM", "ACS_FORM", "ACSA_FORM"],
    "League Position & Gap":     ["HPOS", "APOS", "HPDU", "HPDD", "APDU", "APDD"],
    "Shots (Form)":              ["HS_FORM", "AS_FORM", "HST_FORM", "AST_FORM"],
    "Fouls & Cards (Form)":      ["HF_FORM", "AF_FORM", "HY_FORM", "AY_FORM", "HR_FORM", "AR_FORM"],
    "Corners (Form)":            ["HC_FORM", "AC_FORM"],
}

HEADER_COLOR = "#2c3e50"
ALT_ROW_COLOR = "#f0f3f4"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[NUMERICAL_COLS]
    summary = sub.describe().T.rename(columns={"50%": "median"})
    summary.insert(0, "missing",   sub.isnull().sum())
    summary.insert(1, "missing_%", (sub.isnull().sum() / len(df) * 100).round(2))
    for col in ["mean", "std", "min", "25%", "median", "75%", "max"]:
        summary[col] = summary[col].round(4)
    return summary[["missing", "missing_%", "count", "mean", "std", "min", "25%", "median", "75%", "max"]]


def categorical_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for col in CATEGORICAL_COLS:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = ["value", "count"]
        vc["percentage"] = (vc["count"] / len(df) * 100).round(2)
        result[col] = vc
    return result


def render_df_as_table(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, loc="left")

    col_labels = list(df.columns)
    row_labels = list(df.index.astype(str))
    cell_text = [
        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.values
    ]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d5d8dc")
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#eaecee")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(ALT_ROW_COLOR)
        else:
            cell.set_facecolor("white")


def page_numerical_summary(pdf: PdfPages, df: pd.DataFrame) -> None:
    summary = numerical_summary(df)
    rows_per_page = 28

    for page_start in range(0, len(summary), rows_per_page):
        chunk = summary.iloc[page_start : page_start + rows_per_page]
        fig, ax = plt.subplots(figsize=(16, 10))
        suffix = "" if page_start == 0 else " (cont.)"
        render_df_as_table(ax, chunk, f"Numerical Feature Summary Statistics{suffix}")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def page_categorical_summaries(pdf: PdfPages, df: pd.DataFrame) -> None:
    cat_summaries = categorical_summary(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Categorical Feature Value Counts", fontsize=13, fontweight="bold")

    for ax, (col, vc_df) in zip(axes.flatten(), cat_summaries.items()):
        top = vc_df.head(20).set_index("value")
        render_df_as_table(ax, top, f"{col}  (top {len(top)} of {len(vc_df)})")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    for col in ["HomeTeam", "AwayTeam", "Referee"]:
        vc_df = cat_summaries[col]
        rows_per_page = 30
        for page_start in range(0, len(vc_df), rows_per_page):
            chunk = vc_df.iloc[page_start : page_start + rows_per_page].set_index("value")
            suffix = "" if page_start == 0 else " (cont.)"
            fig, ax = plt.subplots(figsize=(10, 8))
            render_df_as_table(ax, chunk, f"Categorical Summary — {col}{suffix}")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def page_target_distribution(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Target Variable: Full-Time Result (FTR)", fontsize=13, fontweight="bold")

    vc = df[TARGET_COL].value_counts().reindex(["H", "D", "A"])
    colors = [PALETTE[k] for k in vc.index]

    bars = axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar in bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 25,
            f"{int(bar.get_height()):,}",
            ha="center", fontsize=12, fontweight="bold",
        )
    axes[0].set_xlabel("Result", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Absolute Counts", fontsize=11)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(
        ["Home Win (H)", "Draw (D)", "Away Win (A)"], fontsize=10
    )

    pcts = (vc / vc.sum() * 100)
    wedges, _, autotexts = axes[1].pie(
        pcts,
        labels=[f"{k}  {v:.1f}%" for k, v in pcts.items()],
        colors=colors,
        autopct="",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for text in wedges:
        text.set_alpha(0.88)
    axes[1].set_title("Proportions", fontsize=11)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_numerical_distributions(pdf: PdfPages, df: pd.DataFrame) -> None:
    ncols = 4
    cols_per_page = 12

    for page_start in range(0, len(NUMERICAL_COLS), cols_per_page):
        chunk = NUMERICAL_COLS[page_start : page_start + cols_per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4))
        fig.suptitle("Numerical Feature Distributions", fontsize=13, fontweight="bold")
        axes_flat = np.array(axes).flatten()

        for i, col in enumerate(chunk):
            ax = axes_flat[i]
            data = df[col].dropna()
            ax.hist(data, bins=40, color="#3498db", edgecolor="white", alpha=0.78, density=True)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 300)
                ax.plot(xs, kde(xs), color="#1a252f", linewidth=1.4)
            except Exception:
                pass
            ax.axvline(data.mean(),   color="#e74c3c", linestyle="--", linewidth=1.2, label=f"μ={data.mean():.2f}")
            ax.axvline(data.median(), color="#f39c12", linestyle="--", linewidth=1.2, label=f"M={data.median():.2f}")
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.legend(fontsize=6.5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("")

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].axis("off")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def page_correlation_with_target(pdf: PdfPages, df: pd.DataFrame) -> None:
    encoded = df[TARGET_COL].map(TARGET_ENCODING)
    corrs = df[NUMERICAL_COLS].corrwith(encoded).sort_values()

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = [PALETTE["A"] if v < 0 else PALETTE["H"] for v in corrs]
    bars = ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#2c3e50", linewidth=0.9)

    for bar, val in zip(bars, corrs.values):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
        )

    ax.set_xlabel("Pearson Correlation  (FTR encoded: H=2, D=1, A=0)", fontsize=10)
    ax.set_title("Feature Correlation with Target Variable (FTR)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_correlation_heatmap(pdf: PdfPages, df: pd.DataFrame) -> None:
    corr = df[NUMERICAL_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        linewidths=0.25,
        linecolor="#bdc3c7",
        cbar_kws={"shrink": 0.55, "label": "Pearson r"},
    )
    ax.set_title("Inter-Feature Correlation Heatmap (lower triangle)", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _flat_axes(axes: object, n: int) -> List[plt.Axes]:
    arr = np.array(axes).flatten()
    return list(arr[:n])


def page_boxplots_by_target(pdf: PdfPages, df: pd.DataFrame) -> None:
    for group_name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue

        ncols = min(len(present), 4)
        nrows = math.ceil(len(present) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        fig.suptitle(f"Distribution by FTR — {group_name}", fontsize=12, fontweight="bold")
        axes_flat = _flat_axes(axes, len(present))

        order = ["H", "D", "A"]
        for ax, col in zip(axes_flat, present):
            groups = [df[df[TARGET_COL] == o][col].dropna().values for o in order]
            bp = ax.boxplot(groups, patch_artist=True, labels=order, notch=False, widths=0.45)
            for patch, lbl in zip(bp["boxes"], order):
                patch.set_facecolor(PALETTE[lbl])
                patch.set_alpha(0.82)
            for median_line in bp["medians"]:
                median_line.set_color("white")
                median_line.set_linewidth(1.8)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        for ax in _flat_axes(axes, nrows * ncols)[len(present):]:
            ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def page_team_result_profile(pdf: PdfPages, df: pd.DataFrame) -> None:
    team_ftr = (
        df.groupby(["HomeTeam", TARGET_COL])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["H", "D", "A"], fill_value=0)
    )
    team_ftr_pct = team_ftr.div(team_ftr.sum(axis=1), axis=0) * 100
    team_ftr_pct = team_ftr_pct.sort_values("H", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    bottom = np.zeros(len(team_ftr_pct))
    for result in ["H", "D", "A"]:
        ax.barh(
            team_ftr_pct.index,
            team_ftr_pct[result],
            left=bottom,
            color=PALETTE[result],
            label=f"{'Home Win' if result == 'H' else 'Draw' if result == 'D' else 'Away Win'} ({result})",
            edgecolor="white",
        )
        bottom += team_ftr_pct[result].values

    ax.set_xlabel("Percentage of Home Matches (%)", fontsize=11)
    ax.set_title("FTR Distribution by Home Team", fontsize=13, fontweight="bold")
    ax.legend(title="Result", fontsize=9, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_position_analysis(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("League Position at Kick-off vs Match Result", fontsize=13, fontweight="bold")

    order = ["H", "D", "A"]
    for ax, col, label in zip(axes, ["HPOS", "APOS"], ["Home Team Position", "Away Team Position"]):
        groups = [df[df[TARGET_COL] == o][col].dropna().values for o in order]
        bp = ax.boxplot(groups, patch_artist=True, labels=order, notch=False, widths=0.45)
        for patch, lbl in zip(bp["boxes"], order):
            patch.set_facecolor(PALETTE[lbl])
            patch.set_alpha(0.82)
        for median_line in bp["medians"]:
            median_line.set_color("white")
            median_line.set_linewidth(1.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Result", fontsize=10)
        ax.set_ylabel("League Position (1 = Top)", fontsize=10)
        ax.invert_yaxis()

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_derived_goal_differentials(pdf: PdfPages, df: pd.DataFrame) -> None:
    derived: List[Tuple[str, str, str]] = [
        ("HGS",      "AGS",      "Avg Goals Scored Differential (Home − Away)"),
        ("HGC",      "AGC",      "Avg Goals Conceded Differential (Home − Away)"),
        ("HGS_FORM", "AGS_FORM", "Form Goals Scored Differential (Home − Away)"),
        ("HGC_FORM", "AGC_FORM", "Form Goals Conceded Differential (Home − Away)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Derived Goal Differentials vs Match Result", fontsize=13, fontweight="bold")
    order = ["H", "D", "A"]

    for ax, (col_a, col_b, title) in zip(axes.flatten(), derived):
        diff = df[col_a] - df[col_b]
        groups = [diff[df[TARGET_COL] == o].dropna().values for o in order]
        bp = ax.boxplot(groups, patch_artist=True, labels=order, notch=False, widths=0.45)
        for patch, lbl in zip(bp["boxes"], order):
            patch.set_facecolor(PALETTE[lbl])
            patch.set_alpha(0.82)
        for median_line in bp["medians"]:
            median_line.set_color("white")
            median_line.set_linewidth(1.8)
        ax.axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Result", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_form_vs_season_scatter(pdf: PdfPages, df: pd.DataFrame) -> None:
    pairs: List[Tuple[str, str, str]] = [
        ("HPPG", "HPPG_FORM", "Home Points: Season vs Form"),
        ("APPG", "APPG_FORM", "Away Points: Season vs Form"),
        ("HGS",  "HGS_FORM",  "Home Goals Scored: Season vs Form"),
        ("AGS",  "AGS_FORM",  "Away Goals Scored: Season vs Form"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Season Average vs Recent Form (coloured by FTR)", fontsize=13, fontweight="bold")

    for ax, (x_col, y_col, title) in zip(axes.flatten(), pairs):
        for result in ["H", "D", "A"]:
            mask = df[TARGET_COL] == result
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                alpha=0.35,
                s=8,
                color=PALETTE[result],
                label=result,
            )
        lims = [
            min(df[x_col].min(), df[y_col].min()),
            max(df[x_col].max(), df[y_col].max()),
        ]
        ax.plot(lims, lims, "--", color="#7f8c8d", linewidth=0.9, label="y = x")
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, markerscale=2)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_shots_discipline_profile(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Shots & Discipline Form Profiles by FTR", fontsize=13, fontweight="bold")
    order = ["H", "D", "A"]

    pairs = [
        ("HS_FORM",  "AS_FORM",  "Shots"),
        ("HST_FORM", "AST_FORM", "Shots on Target"),
        ("HC_FORM",  "AC_FORM",  "Corners"),
        ("HF_FORM",  "AF_FORM",  "Fouls"),
        ("HY_FORM",  "AY_FORM",  "Yellow Cards"),
        ("HR_FORM",  "AR_FORM",  "Red Cards"),
    ]

    for ax, (home_col, away_col, label) in zip(axes.flatten(), pairs):
        h_means = [df[df[TARGET_COL] == o][home_col].mean() for o in order]
        a_means = [df[df[TARGET_COL] == o][away_col].mean() for o in order]

        x = np.arange(3)
        width = 0.35
        ax.bar(x - width / 2, h_means, width, label="Home", color="#3498db", edgecolor="white", alpha=0.88)
        ax.bar(x + width / 2, a_means, width, label="Away", color="#e67e22", edgecolor="white", alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        ax.set_title(f"Avg {label} Form", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_title_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    fig.patch.set_facecolor("#1a252f")

    ax.text(0.5, 0.88, "Premier League Match Dataset",
            ha="center", va="center", fontsize=22, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.78, "Exploratory Data Analysis Report",
            ha="center", va="center", fontsize=16, color="#aab7b8", transform=ax.transAxes)

    ax.axhline(0.72, xmin=0.15, xmax=0.85, color="#3498db", linewidth=1.5)

    stats = [
        ("Total Matches",         f"{len(df):,}"),
        ("Total Features",        str(len(df.columns))),
        ("Numerical Features",    str(len(NUMERICAL_COLS))),
        ("Categorical Features",  str(len(CATEGORICAL_COLS))),
        ("Total Missing Values",  str(int(df.isnull().sum().sum()))),
        ("Home Wins  (H)",        f"{(df[TARGET_COL] == 'H').sum():,}  ({(df[TARGET_COL] == 'H').mean()*100:.1f}%)"),
        ("Draws      (D)",        f"{(df[TARGET_COL] == 'D').sum():,}  ({(df[TARGET_COL] == 'D').mean()*100:.1f}%)"),
        ("Away Wins  (A)",        f"{(df[TARGET_COL] == 'A').sum():,}  ({(df[TARGET_COL] == 'A').mean()*100:.1f}%)"),
    ]

    for i, (label, value) in enumerate(stats):
        y = 0.63 - i * 0.07
        ax.text(0.30, y, label + ":", ha="right", va="center", fontsize=12,
                color="#aab7b8", transform=ax.transAxes)
        ax.text(0.33, y, value,         ha="left",  va="center", fontsize=12,
                color="white",   fontweight="bold", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_path: str) -> None:
    with PdfPages(output_path) as pdf:
        add_title_page(pdf, df)
        page_numerical_summary(pdf, df)
        page_categorical_summaries(pdf, df)
        page_target_distribution(pdf, df)
        page_numerical_distributions(pdf, df)
        page_correlation_with_target(pdf, df)
        page_correlation_heatmap(pdf, df)
        page_boxplots_by_target(pdf, df)
        page_team_result_profile(pdf, df)
        page_position_analysis(pdf, df)
        page_derived_goal_differentials(pdf, df)
        page_form_vs_season_scatter(pdf, df)
        page_shots_discipline_profile(pdf, df)


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in NUMERICAL_COLS + CATEGORICAL_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive EDA report for the enriched Premier League dataset."
    )
    parser.add_argument("--input",  required=True, help="Path to the enriched input CSV.")
    parser.add_argument("--output", required=True, help="Path for the EDA output PDF.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validate_input(df)
    run_eda(df, args.output)


if __name__ == "__main__":
    main()