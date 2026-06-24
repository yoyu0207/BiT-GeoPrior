import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an 8-channel Sentinel-2 stack: B8/B4/B3/B2/NDVI/EVI/SAVI/GNDVI."
    )
    parser.add_argument("--input", required=True, help="4-band S2 mosaic in B8/B4/B3/B2 order.")
    parser.add_argument("--output", required=True, help="Output 8-channel GeoTIFF.")
    parser.add_argument("--block", type=int, default=1024)
    return parser.parse_args()


def calculate_indices(s2):
    eps = 1e-6
    nir = s2[0]
    red = s2[1]
    green = s2[2]
    blue = s2[3]
    ndvi = (nir - red) / (nir + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    gndvi = (nir - green) / (nir + green + eps)
    return [ndvi, evi, savi, gndvi]


def clean_reflectance(arr):
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < -100] = 0.0
    if float(np.nanmax(arr)) > 100:
        arr = arr / 10000.0
    return arr


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if src.count < 4:
            raise ValueError("Input mosaic must contain at least 4 bands in B8/B4/B3/B2 order.")
        profile = src.profile.copy()
        profile.update(count=8, dtype="float32", compress="lzw", nodata=None)

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = []
            for row in range(0, src.height, args.block):
                for col in range(0, src.width, args.block):
                    width = min(args.block, src.width - col)
                    height = min(args.block, src.height - row)
                    windows.append(Window(col, row, width, height))

            for window in tqdm(windows, desc="Build 8ch stack"):
                s2 = clean_reflectance(src.read([1, 2, 3, 4], window=window))
                stack = np.stack(list(s2) + calculate_indices(s2)).astype(np.float32)
                stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
                dst.write(stack, window=window)

            names = ["B8_NIR", "B4_Red", "B3_Green", "B2_Blue", "NDVI", "EVI", "SAVI", "GNDVI"]
            for idx, name in enumerate(names, start=1):
                dst.set_band_description(idx, name)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
