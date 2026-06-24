from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:/yoyu/SA_Identification")
FIG_DIR = ROOT / "project" / "fig"

YEAR_FILES = {
    2023: ROOT / "Results_rebuilt" / "2020_2023_GOEP" / "2020_2023_area_stats.csv",
    2024: ROOT / "Results_rebuilt" / "2020_2024_GOEP" / "2020_2024_area_stats.csv",
    2025: ROOT / "Results_rebuilt" / "2020_2025_GOEP" / "2020_2025__area_stats.csv",
}

ESTUARY_ABBR = {
    "川东港": "CDG",
    "弶港区间": "JG",
    "串水港": "CSG",
    "绣针河口": "XZH",
    "王港": "WG",
    "梁剁河口": "LDH",
    "临洪口": "LHK",
    "新洋港": "XYG",
    "中路港": "ZLG",
    "斗龙港": "DLG",
    "小洋口": "XYK",
    "东凌港": "DLH",
    "通州湾": "TZW",
}

STACK_COLORS = [
    "#DB7036",
    "#ED8D5A",
    "#EC9E58",
    "#DBCB92",
    "#7BC0CD",
    "#4198AC",
    "#51999F",
    "#BFDFD2",
    "#AFAFAF",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows: list[dict] = []
    long_rows: list[pd.DataFrame] = []

    for year, csv_path in YEAR_FILES.items():
        df = pd.read_csv(csv_path)
        df["area_ha"] = df["area_ha"].astype(float)
        df["Year"] = year
        df["abbr"] = df["Name"].map(ESTUARY_ABBR).fillna("OTH")
        long_rows.append(df[["Year", "Name", "abbr", "area_ha"]].copy())
        yearly_rows.append(
            {
                "Year": year,
                "Area_ha": df["area_ha"].sum(),
                "Area_kha": df["area_ha"].sum() / 1000.0,
            }
        )

    totals = pd.DataFrame(yearly_rows).sort_values("Year").reset_index(drop=True)
    long_df = pd.concat(long_rows, ignore_index=True)
    return totals, long_df


def build_composition(long_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    ranking = (
        long_df.groupby(["abbr"], as_index=False)["area_ha"]
        .sum()
        .sort_values("area_ha", ascending=False)
    )
    top_abbr = ranking.head(top_n)["abbr"].tolist()
    pivot = (
        long_df.pivot_table(index="Year", columns="abbr", values="area_ha", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    ordered = [abbr for abbr in top_abbr if abbr in pivot.columns and abbr != "OTH"]
    remainder = [c for c in pivot.columns if c not in ordered and c != "OTH"]
    other_series = pd.Series(0.0, index=pivot.index)
    if "OTH" in pivot.columns:
        other_series = other_series.add(pivot["OTH"], fill_value=0.0)
    if remainder:
        other_series = other_series.add(pivot[remainder].sum(axis=1), fill_value=0.0)
        pivot = pivot.drop(columns=remainder)
    pivot["OTH"] = other_series
    ordered.append("OTH")
    return pivot.reindex(columns=ordered).reset_index()


def export_tables(totals: pd.DataFrame, composition: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    totals_out = totals.copy()
    totals_out["Area_ha"] = totals_out["Area_ha"].round(2)
    totals_out["Area_kha"] = totals_out["Area_kha"].round(3)
    composition_out = composition.copy()
    for col in composition_out.columns:
        if col != "Year":
            composition_out[col] = composition_out[col].round(2)
    totals_out.to_csv(
        FIG_DIR / "yearly_eradication_totals_2023_2025.csv",
        index=False,
        float_format="%.3f",
    )
    composition_out.to_csv(
        FIG_DIR / "yearly_eradication_composition_2023_2025.csv",
        index=False,
        float_format="%.2f",
    )


def plot_figure(totals: pd.DataFrame, composition: pd.DataFrame) -> None:
    years = totals["Year"].to_numpy()
    values = totals["Area_kha"].to_numpy()

    fig = plt.figure(figsize=(8.6, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 0.8], wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    bar_colors = ["#8FA7BF", "#4F7198", "#2E556E"]
    x_pos1 = np.array([0.00, 0.80, 1.60])
    bars = ax1.bar(x_pos1, values, color=bar_colors, width=0.46, edgecolor="white", linewidth=0.8)
    ax1.plot(x_pos1, values, color="#3E4C59", lw=1.2, marker="o", ms=3.5, zorder=3)
    ax1.set_xticks(x_pos1, years.astype(str))
    ax1.set_ylabel(r"Eradication area ($\times 10^3$ ha)")
    ax1.set_title("(a) Total eradication area", loc="left", pad=4, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", color="#D4D9DE", linewidth=0.7)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for rect, val in zip(bars, values):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.18,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2E3A45",
        )

    x_pos2 = np.array([0.00, 0.4, 0.8])
    bottoms = np.zeros(len(composition), dtype=float)
    columns = [c for c in composition.columns if c != "Year"]
    for idx, col in enumerate(columns):
        vals = composition[col].to_numpy() / 1000.0
        ax2.bar(
            x_pos2,
            vals,
            bottom=bottoms,
            color=STACK_COLORS[idx % len(STACK_COLORS)],
            width=0.22,
            edgecolor="white",
            linewidth=0.45,
            label=col,
        )
        bottoms += vals
    ax2.set_xticks(x_pos2, composition["Year"].astype(str).tolist())
    ax2.set_ylabel(r"Eradication area ($\times 10^3$ ha)")
    ax2.set_title("(b) Major-estuary composition", loc="left", pad=4, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", color="#D4D9DE", linewidth=0.7)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.legend(
        ncol=4,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.10),
        handlelength=1.1,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.28)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_yearly_eradication_statistics.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_yearly_eradication_statistics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_yearly_eradication_statistics.tif", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    totals, long_df = load_tables()
    composition = build_composition(long_df, top_n=8)
    export_tables(totals, composition)
    plot_figure(totals, composition)


if __name__ == "__main__":
    main()
