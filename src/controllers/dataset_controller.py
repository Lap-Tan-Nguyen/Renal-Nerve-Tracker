import os
from typing import Any, Dict, List, cast
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from io.exporter import append_table_to_dataset


class DatasetController:
    """Coordinates summary stats, global exports, and high-res overlay images."""

    def __init__(self, dataset_path: str, reconstruction_dir: str) -> None:
        self.dataset_path: str = dataset_path
        self.reconstruction_dir: str = reconstruction_dir

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Reads current training dataset and returns total object counts[cite: 3, 5]."""
        if not os.path.exists(self.dataset_path):
            return {"Total Objects": 0, "Selected Nerves": 0, "Rejected Objects": 0}

        df: pd.DataFrame = pd.read_excel(self.dataset_path)
        if df.empty or "IsSelected" not in df.columns:
            return {"Total Objects": 0, "Selected Nerves": 0, "Rejected Objects": 0}

        total_objs: int = len(df)
        selected: int = int((df["IsSelected"] == True).sum())
        rejected: int = total_objs - selected

        return {
            "Total Objects": total_objs,
            "Selected Nerves": selected,
            "Rejected Objects": rejected,
        }

    def export_overlay_tiff(
        self,
        image_name: str,
        img_full: NDArray[np.float64],
        output_dir: str = "overlay_exports",
    ) -> str:
        """Generates high-resolution TIFF overlay of annotated nerve objects[cite: 14]."""
        import scipy.ndimage as ndi
        from skimage.io import imsave

        os.makedirs(output_dir, exist_ok=True)
        base_name: str = os.path.splitext(image_name)[0]
        out_tiff_path: str = os.path.join(output_dir, f"{base_name}_nerve_overlay.tif")

        # Composite full mask from .npz reconstruction files[cite: 13, 14]
        full_mask: NDArray[np.bool_] = np.zeros(
            (img_full.shape[0], img_full.shape[1]), dtype=bool
        )

        for npz_file in os.listdir(self.reconstruction_dir):
            if npz_file.startswith(base_name) and npz_file.endswith(".npz"):
                data: np.lib.npyio.NpzFile = np.load(
                    os.path.join(self.reconstruction_dir, npz_file)
                )
                x1, y1 = int(data["x1"]), int(data["y1"])
                x2, y2 = int(data["x2"]), int(data["y2"])
                roi_mask: NDArray[np.bool_] = cast(NDArray[np.bool_], data["selected_mask"])

                full_mask[y1 : y2 + 1, x1 : x2 + 1] |= roi_mask

        # Generate red overlay channel[cite: 14]
        rgb_img: NDArray[np.uint8] = (np.stack([img_full] * 3, axis=-1) * 255).astype(
            np.uint8
        )
        rgb_img[full_mask, 0] = 255  # Set red channel to max
        rgb_img[full_mask, 1] = 0
        rgb_img[full_mask, 2] = 0

        imsave(out_tiff_path, rgb_img)
        print(f"Saved high-res overlay TIFF to: {out_tiff_path}")
        return out_tiff_path