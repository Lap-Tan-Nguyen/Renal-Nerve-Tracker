import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union


def build_auto_roi_grid(
    img_size: Tuple[int, ...], approx_num_rois: int = 9
) -> pd.DataFrame:
    """Build near-square ROI grid and return DataFrame with 0-indexed ROI bounds[cite: 2]."""
    if len(img_size) < 2:
        raise ValueError("img_size must be at least (rows, cols).")

    nrows_img, ncols_img = img_size[0], img_size[1]

    if approx_num_rois is None or approx_num_rois < 1:
        approx_num_rois = 9

    approx_num_rois = int(round(approx_num_rois))

    grid_rows = int(np.floor(np.sqrt(approx_num_rois)))
    grid_cols = int(np.ceil(approx_num_rois / grid_rows))

    while grid_rows * grid_cols < approx_num_rois:
        grid_rows += 1

    actual_num_rois = grid_rows * grid_cols

    row_edges = np.round(np.linspace(0, nrows_img, grid_rows + 1)).astype(int)
    col_edges = np.round(np.linspace(0, ncols_img, grid_cols + 1)).astype(int)

    roi_records: List[Dict[str, Union[int, float]]] = []
    roi_id = 0

    for r in range(grid_rows):
        for c in range(grid_cols):
            roi_id += 1
            x1 = int(col_edges[c])
            x2 = int(col_edges[c + 1] - 1)
            y1 = int(row_edges[r])
            y2 = int(row_edges[r + 1] - 1)

            roi_records.append(
                {
                    "ROI_ID": roi_id,
                    "ROI_Row": r + 1,
                    "ROI_Col": c + 1,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                }
            )

    roi_table = pd.DataFrame(roi_records)
    print(f"Requested ~{approx_num_rois} ROIs.")
    print(
        f"Generated grid: {grid_rows} rows x {grid_cols} cols = {actual_num_rois} total ROIs."
    )

    return roi_table