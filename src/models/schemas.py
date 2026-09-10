from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ROIMeta:
    """Metadata for an ROI tile[cite: 2]."""

    image_name: str
    roi_id: int
    roi_row: int
    roi_col: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ArteryParams:
    """Configuration parameters for artery segmentation[cite: 15, 19]."""

    use_clahe: bool = True
    clahe_clip_limit: float = 0.01
    gaussian_sigma: float = 1.0
    grad_floor: float = 0.05
    use_otsu_threshold: bool = True
    otsu_scale_factor: float = 1.15
    min_blob_area: int = 800
    close_radius: int = 2
    fill_dilate_radius: int = 1


@dataclass
class ArteryResult:
    """Data and mask outputs resulting from artery segmentation[cite: 12, 19]."""

    params: ArteryParams
    roi_pos: Tuple[int, int, int, int]  # (x, y, width, height)[cite: 19]
    img_roi: NDArray[np.float64]
    img_pre: NDArray[np.float64]
    bw_final: NDArray[np.bool_]
    bw_filled: NDArray[np.bool_]
    centroid_roi: Tuple[float, float]
    mask_full: NDArray[np.bool_]
    img_full_no_artery: NDArray[np.float64]
    filled_gain: int


@dataclass
class NerveParams:
    """Parameters for dark nerve candidate segmentation[cite: 16]."""

    dark_threshold: float = 0.35
    area_cutoff: int = 50
    connect_radius: int = 2


@dataclass
class NerveSegmentationResult:
    """Outputs generated during ROI nerve candidate segmentation[cite: 16]."""

    params: NerveParams
    roi_img: NDArray[np.float64]
    bw_dark: NDArray[np.bool_]
    bw_keep: NDArray[np.bool_]
    bw_merged: NDArray[np.bool_]
    bw_filled: NDArray[np.bool_]
    labeled_filled: NDArray[np.int32]
    num_objects: int