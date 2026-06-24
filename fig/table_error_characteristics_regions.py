from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


ROOT = Path(r"D:/yoyu/SA_Identification")
OUT_DIR = ROOT / "project" / "fig"

PRED_PATH = ROOT / "Results" / "Prediction_2020_2024_Governance0330.tif"
REF_PATH = ROOT / "Dataset" / "label" / "label001.tif"

REGIONS = [
    ("Region I", "large irregular patch", 12000, 16400, 1600),
    ("Region II", "linear tidal-flat edge", 0, 2100, 1400),
    ("Region III", "fragmented tidal-channel landscape", 15200, 26800, 1600),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def compute_table() -> pd.DataFrame:
    rows = []
    with rasterio.open(PRED_PATH) as pred_src, rasterio.open(REF_PATH) as ref_src:
        pixel_area_ha = abs(pred_src.transform.a * pred_src.transform.e) / 10000.0
        for region, pattern, x, y, size in REGIONS:
            window = Window(x, y, size, size)
            pred = pred_src.read(1, window=window) == 1
            ref = ref_src.read(1, window=window) == 1

            tp = int(np.logical_and(pred, ref).sum())
            fp = int(np.logical_and(pred, ~ref).sum())
            fn = int(np.logical_and(~pred, ref).sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            omission = fn / (tp + fn) if (tp + fn) else 0.0
            commission = fp / (tp + fp) if (tp + fp) else 0.0

            if commission - omission > 0.03:
                error_flag = "FP-dominant"
            elif omission - commission > 0.03:
                error_flag = "FN-dominant"
            else:
                error_flag = "balanced"

            rows.append(
                {
                    "Region": region,
                    "Pattern": pattern,
                    "TP (ha)": tp * pixel_area_ha,
                    "FP (ha)": fp * pixel_area_ha,
                    "FN (ha)": fn * pixel_area_ha,
                    "Precision (%)": precision * 100,
                    "Recall (%)": recall * 100,
                    "F1 (%)": f1 * 100,
                    "Commission error (%)": commission * 100,
                    "Omission error (%)": omission * 100,
                    "Dominant characteristic": error_flag,
                }
            )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "TP (ha)",
        "FP (ha)",
        "FN (ha)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "Commission error (%)",
        "Omission error (%)",
    ]
    for col in numeric_cols:
        out[col] = out[col].map(lambda v: f"{v:.2f}")
    return out


def render_table(df_fmt: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.8, 2.6))
    ax.axis("off")

    columns = [
        "Region",
        "Pattern",
        "TP (ha)",
        "FP (ha)",
        "FN (ha)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "Commission error (%)",
        "Omission error (%)",
        "Dominant characteristic",
    ]
    header_labels = [
        "Region",
        "Pattern",
        "TP\n(ha)",
        "FP\n(ha)",
        "FN\n(ha)",
        "Precision\n(%)",
        "Recall\n(%)",
        "F1\n(%)",
        "Commission\nerror (%)",
        "Omission\nerror (%)",
        "Dominant\ncharacteristic",
    ]
    cell_text = df_fmt[columns].values.tolist()

    col_widths = [0.085, 0.215, 0.068, 0.068, 0.068, 0.082, 0.076, 0.068, 0.095, 0.095, 0.11]
    table = ax.table(
        cellText=cell_text,
        colLabels=header_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.15)
    table.scale(1, 1.48)

    header_color = "#E8EEF7"
    row_alt = "#F8FAFC"
    edge = "#AEB8C2"
    text_dark = "#233142"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(text_dark)
            cell.get_text().set_fontsize(7.0)
        else:
            cell.set_facecolor(row_alt if row % 2 == 1 else "white")
            if col in (1, 10):
                cell._loc = "left"
                cell.PAD = 0.012

    fig.text(
        0.012,
        0.96,
        "Table X. Error characteristics of G-OEP-BiT in representative tidal-flat regions.",
        ha="left",
        va="top",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.06,
        "FP: false positive; FN: false negative. Areas were calculated at 10 m pixel resolution.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#4E5B68",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "table_error_characteristics_regions.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "table_error_characteristics_regions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = compute_table()
    df_fmt = format_table(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "table_error_characteristics_regions.csv", index=False)
    df_fmt.to_markdown(OUT_DIR / "table_error_characteristics_regions.md", index=False)
    render_table(df_fmt)


if __name__ == "__main__":
    main()
