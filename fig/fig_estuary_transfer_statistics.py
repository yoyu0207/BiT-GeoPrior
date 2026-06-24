from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


ROOT = Path(r"D:\yoyu\SA_Identification")
FIG_DIR = ROOT / "project" / "fig"
STUDYAREA_PATH = ROOT / "StudyArea" / "StudyArea.shp"
REFERENCE_LABEL_PATH = ROOT / "Dataset" / "label" / "label_2020_2024.shp"

# Direct-transfer outputs. Replace or extend these paths when newer yearly maps are ready.
TRANSFER_RASTERS = {
    "2019-2023": ROOT / "Results" / "Prediction_2019_2023_Governance2.tif",
    "2020-2024": ROOT / "Results" / "Prediction_2020_2024_Governance0330.tif",
    "2022-2023": ROOT / "Results" / "Prediction_2022_2023_Governance2.tif",
    "2022-2024": ROOT / "Results" / "Prediction_2022_2024_Governance2.tif",
    "2022-2025": ROOT / "Results" / "Prediction_2022_2025_Governance2.tif",
}

TOP_N = 8
OUT_PREFIX = FIG_DIR / "fig_estuary_transfer_statistics"

# ASCII abbreviations keep the figure compatible with Times New Roman.
ESTUARY_LABELS = {
    1: "XZH",
    2: "ZPK",
    3: "LHK",
    4: "LZK",
    5: "GHK",
    6: "XHH",
    7: "SYH",
    8: "XYG",
    9: "ZLG",
    10: "DLG",
    11: "SYY",
    12: "WG",
    13: "CDG",
    14: "CSG",
    15: "LDH",
    16: "JG",
    17: "XYK",
    18: "LB",
    19: "YKG",
    20: "DNG",
    21: "TZW",
    22: "DYG",
    23: "HZG",
    24: "TLG",
    25: "XXG",
    26: "OTH",
}

ESTUARY_NAMES_EN = {
    1: "Xiuzhen River Estuary",
    2: "Zhupeng Interval",
    3: "Linhong Estuary",
    4: "Lezi Estuary",
    5: "Guanhe Estuary",
    6: "Xinhuai Interval",
    7: "Sheyang River Estuary",
    8: "Xinyang Harbor",
    9: "Zhonglu Harbor",
    10: "Doulong Harbor",
    11: "Siyinyou Sluice",
    12: "Wanggang Harbor",
    13: "Chuandong Harbor",
    14: "Chuanshui Harbor",
    15: "Liangduo River Estuary",
    16: "Jiangang Interval",
    17: "Xiaoyangkou",
    18: "Liubu Harbor",
    19: "Yangkou Harbor",
    20: "Dongling Harbor",
    21: "Tongzhou Bay",
    22: "Dayang Harbor",
    23: "Haozhi Harbor",
    24: "Tanglu Harbor",
    25: "Xiexing Harbor",
    26: "Other",
}

STACK_COLORS = [
    "#264653",
    "#2a9d8f",
    "#5e8c61",
    "#6d597a",
    "#b56576",
    "#e56b6f",
    "#eaac8b",
    "#f4a261",
    "#c9c9c9",
]


def set_matplotlib_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compute_reference_area_by_estuary(study_gdf, label_path):
    labels = gpd.read_file(label_path)
    labels = labels[labels["value"] == 1].copy().to_crs(study_gdf.crs)
    inter = gpd.overlay(
        study_gdf[["ID", "Name", "geometry"]],
        labels[["geometry"]],
        how="intersection",
    )
    inter["reference_area_ha"] = inter.geometry.area / 10000
    summary = (
        inter.groupby(["ID", "Name"], as_index=False)["reference_area_ha"]
        .sum()
        .sort_values("reference_area_ha", ascending=False)
    )
    return summary


def zonal_positive_area(raster_path, study_gdf, positive_value=1):
    rows = []
    with rasterio.open(raster_path) as src:
        pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000
        zones = study_gdf.to_crs(src.crs)
        for _, zone in zones.iterrows():
            minx, miny, maxx, maxy = zone.geometry.bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            mask = geometry_mask(
                [zone.geometry],
                out_shape=data.shape,
                transform=transform,
                invert=True,
            )
            pixel_count = np.count_nonzero((data == positive_value) & mask)
            rows.append(
                {
                    "ID": int(zone["ID"]),
                    "Name": zone["Name"],
                    "predicted_area_ha": pixel_count * pixel_area_ha,
                }
            )
    return pd.DataFrame(rows)


def build_statistics():
    study = gpd.read_file(STUDYAREA_PATH)
    study["Name"] = study["Name"].astype(str).str.strip()
    study = study[study["Name"] != "其他"].copy()

    reference_df = compute_reference_area_by_estuary(study, REFERENCE_LABEL_PATH)
    top_ids = reference_df.head(TOP_N)["ID"].tolist()

    reference_df["abbr"] = reference_df["ID"].map(ESTUARY_LABELS)
    reference_df["name_en"] = reference_df["ID"].map(ESTUARY_NAMES_EN)

    transfer_rows = []
    for period, raster_path in TRANSFER_RASTERS.items():
        df = zonal_positive_area(raster_path, study, positive_value=1)
        df["period"] = period
        transfer_rows.append(df)
    transfer_df = pd.concat(transfer_rows, ignore_index=True)
    transfer_df["abbr"] = transfer_df["ID"].map(ESTUARY_LABELS)
    transfer_df["name_en"] = transfer_df["ID"].map(ESTUARY_NAMES_EN)

    top_reference = reference_df[reference_df["ID"].isin(top_ids)].copy()
    top_reference["rank"] = top_reference["reference_area_ha"].rank(
        method="first", ascending=False
    )
    top_reference = top_reference.sort_values("rank")

    top_transfer = transfer_df[transfer_df["ID"].isin(top_ids)].copy()
    top_transfer["rank"] = top_transfer["ID"].map(
        dict(zip(top_reference["ID"], range(len(top_reference))))
    )
    top_transfer = top_transfer.sort_values(["period", "rank"])

    pred_2020 = (
        transfer_df[transfer_df["period"] == "2020-2024"][["ID", "predicted_area_ha"]]
        .rename(columns={"predicted_area_ha": "predicted_2020_2024_ha"})
    )
    top_compare = (
        top_reference.merge(pred_2020, on="ID", how="left")
        .sort_values("rank")
        .reset_index(drop=True)
    )

    period_totals = (
        transfer_df.groupby("period", as_index=False)["predicted_area_ha"]
        .sum()
        .rename(columns={"predicted_area_ha": "total_predicted_area_ha"})
    )

    composition = (
        top_transfer.pivot_table(
            index="period", columns="abbr", values="predicted_area_ha", aggfunc="sum"
        )
        .fillna(0.0)
        .reindex(TRANSFER_RASTERS.keys())
    )
    transfer_pivot = (
        transfer_df.pivot_table(
            index="period", columns="ID", values="predicted_area_ha", aggfunc="sum"
        )
        .fillna(0.0)
        .reindex(TRANSFER_RASTERS.keys())
    )
    composition["Others"] = (
        transfer_pivot.drop(columns=top_ids, errors="ignore").sum(axis=1).values
    )

    percentage = composition.div(composition.sum(axis=1).replace(0, np.nan), axis=0) * 100
    percentage = percentage.fillna(0.0)

    return top_compare, period_totals, composition, percentage, reference_df, transfer_df


def plot_figure(top_compare, period_totals, percentage):
    fig = plt.figure(figsize=(12.6, 5.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.12], wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    labels = top_compare["abbr"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    ref_vals = top_compare["reference_area_ha"].to_numpy() / 1000
    pred_vals = top_compare["predicted_2020_2024_ha"].to_numpy() / 1000

    ax1.bar(
        x - width / 2,
        ref_vals,
        width=width,
        color="#355f93",
        edgecolor="white",
        linewidth=0.8,
        label="Reference area",
    )
    ax1.bar(
        x + width / 2,
        pred_vals,
        width=width,
        color="#efc1a6",
        edgecolor="white",
        linewidth=0.8,
        label="Predicted area",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"Area ($\times$ 10$^3$ ha)")
    ax1.set_title("(a)", loc="left", fontsize=18, pad=2)
    ax1.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, color="#d7d7d7")
    ax1.set_axisbelow(True)
    ax1.legend(
        frameon=False,
        loc="upper right",
        handlelength=0.8,
        handletextpad=0.4,
    )

    stack_order = percentage.columns.tolist()
    periods = list(percentage.index)
    xpos = np.arange(len(periods))
    bottoms = np.zeros(len(periods))
    for idx, estuary in enumerate(stack_order):
        vals = percentage[estuary].to_numpy()
        ax2.bar(
            xpos,
            vals,
            bottom=bottoms,
            width=0.64,
            color=STACK_COLORS[idx % len(STACK_COLORS)],
            edgecolor="none",
            label=estuary,
        )
        bottoms += vals

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(periods)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Area percentage (%)")
    ax2.set_title("(b)", loc="left", fontsize=18, pad=2)
    ax2.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, color="#d7d7d7")
    ax2.set_axisbelow(True)

    ax2r = ax2.twinx()
    total_vals = (
        period_totals.set_index("period").loc[periods, "total_predicted_area_ha"].to_numpy()
        / 1000
    )
    ax2r.plot(
        xpos,
        total_vals,
        color="black",
        linewidth=1.4,
        marker="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax2r.set_ylabel(r"Total area ($\times$ 10$^3$ ha)")

    for x0, y0 in zip(xpos, total_vals):
        ax2r.text(
            x0,
            y0 + max(total_vals.max() * 0.025, 0.12),
            f"{y0:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.legend(
        ncol=3,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        columnspacing=0.8,
        handlelength=0.8,
        handletextpad=0.3,
    )

    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.95)
    return fig


def save_outputs(top_compare, period_totals, composition, percentage, reference_df, transfer_df):
    top_compare.to_csv(FIG_DIR / "estuary_area_statistics_top8.csv", index=False, encoding="utf-8-sig")
    period_totals.to_csv(FIG_DIR / "transfer_period_totals.csv", index=False, encoding="utf-8-sig")
    composition.to_csv(FIG_DIR / "transfer_period_composition_top8.csv", encoding="utf-8-sig")
    percentage.to_csv(FIG_DIR / "transfer_period_percentage_top8.csv", encoding="utf-8-sig")
    reference_df.to_csv(FIG_DIR / "estuary_reference_area_all.csv", index=False, encoding="utf-8-sig")
    transfer_df.to_csv(FIG_DIR / "estuary_transfer_area_all.csv", index=False, encoding="utf-8-sig")


def main():
    set_matplotlib_style()
    top_compare, period_totals, composition, percentage, reference_df, transfer_df = build_statistics()
    save_outputs(top_compare, period_totals, composition, percentage, reference_df, transfer_df)
    fig = plot_figure(top_compare, period_totals, percentage)
    for suffix in [".png", ".pdf", ".tif"]:
        fig.savefig(OUT_PREFIX.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUT_PREFIX.with_suffix('.png')}")


if __name__ == "__main__":
    main()
