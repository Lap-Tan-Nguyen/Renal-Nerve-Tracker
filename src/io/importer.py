import os
from typing import Dict, List, Optional, Tuple, Any, cast
import numpy as np
from numpy.typing import NDArray
from skimage.io import imread


def load_histology_image(file_path: str) -> NDArray[np.float64]:
    """Loads a histology image slice and converts it to a normalized float64 array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    raw_img: NDArray[Any] = imread(file_path)
    img_float: NDArray[np.float64] = raw_img.astype(np.float64)

    if float(np.max(img_float)) > 1.0:
        img_float = img_float / 255.0

    return img_float


def find_completed_roi_ids(reconstruction_dir: str, image_name: str) -> List[int]:
    """Scans reconstruction directory for existing ROI session files to support resuming progress[cite: 3, 13]."""
    if not os.path.exists(reconstruction_dir):
        return []

    base_name: str = os.path.splitext(image_name)[0]
    completed_ids: List[int] = []

    for file_name in os.listdir(reconstruction_dir):
        if file_name.startswith(base_name) and file_name.endswith(".npz"):
            try:
                # File format: <base>_ROI_<id>_reconstruction.npz[cite: 13]
                parts: List[str] = file_name.split("_ROI_")
                if len(parts) > 1:
                    roi_id_str: str = parts[1].split("_")[0]
                    completed_ids.append(int(roi_id_str))
            except ValueError:
                continue

    return sorted(completed_ids)