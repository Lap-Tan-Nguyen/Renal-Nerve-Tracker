import os
from typing import Optional, Union
import numpy as np
import pandas as pd

from models.schemas import ROIMeta


def append_table_to_dataset(
    new_df: pd.DataFrame, file_path: str, sheet_name: str = "Sheet1"
) -> None:
    """Appends DataFrame rows to an existing Excel/CSV dataset.

    Harmonizes missing columns dynamically (equivalent to appendTableToExcel.m)[cite: 1].
    """
    if new_df.empty:
        return

    file_ext: str = os.path.splitext(file_path)[1].lower()

    if os.path.exists(file_path):
        if file_ext in [".xlsx", ".xls"]:
            existing_df: pd.DataFrame = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            existing_df = pd.read_csv(file_path)

        # Concatenation automatically harmonizes schema mismatches by filling NaNs[cite: 1]
        combined_df: pd.DataFrame = pd.concat(
            [existing_df, new_df], ignore_index=True, sort=False
        )
    else:
        combined_df = new_df

    # Write back harmonized output
    if file_ext in [".xlsx", ".xls"]:
        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="w"
        ) as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        combined_df.to_csv(file_path, index=False)


def save_roi_reconstruction_data(
    image_name: str,
    roi_meta: ROIMeta,
    roi_img: np.ndarray,
    bw_selected: np.ndarray,
    full_image_shape: tuple[int, int],
    out_folder: str,
) -> str:
    """Saves per-ROI segmentation reconstruction data to NumPy compressed format (.npz)[cite: 13].

    Translates saveROIReconstructionData.m[cite: 13].
    """
    os.makedirs(out_folder, exist_ok=True)
    base_name: str = os.path.splitext(image_name)[0]
    out_path: str = os.path.join(
        out_folder, f"{base_name}_ROI_{roi_meta.roi_id:03d}_reconstruction.npz"
    )

    np.savez_compressed(
        out_path,
        image_name=image_name,
        roi_id=roi_meta.roi_id,
        roi_row=roi_meta.roi_row,
        roi_col=roi_meta.roi_col,
        x1=roi_meta.x1,
        y1=roi_meta.y1,
        x2=roi_meta.x2,
        y2=roi_meta.y2,
        full_image_rows=full_image_shape[0],
        full_image_cols=full_image_shape[1],
        roi_image=roi_img,
        selected_mask=bw_selected,
    )

    return out_path