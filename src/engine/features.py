from typing import Any, Dict, List, cast
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.morphology.binary import binary_erosion

from models.schemas import NerveSegmentationResult, ROIMeta


def extract_nerve_candidate_features(
    roi_img: NDArray[np.float64],
    seg_result: NerveSegmentationResult,
    bw_selected: NDArray[np.bool_],
    roi_meta: ROIMeta,
    selection_source: str = "auto_label",
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """Extracts morphological and boundary gradient features for candidates[cite: 5]."""
    if seg_result.num_objects == 0:
        return pd.DataFrame()

    if roi_img.ndim == 3:
        roi_img = roi_img.mean(axis=2)

    g_mag: NDArray[np.float64] = sobel(roi_img)
    regions = regionprops(seg_result.labeled_filled, intensity_image=roi_img)

    records: List[Dict[str, Any]] = []
    eps: float = 1e-8

    raw_erosion_disk: Any = cast(Any, disk)(1)
    erosion_footprint: NDArray[np.uint8] = np.asarray(
        raw_erosion_disk, dtype=np.uint8
    )

    for k, prop in enumerate(regions, start=1):
        obj_mask_filled: NDArray[np.bool_] = (
            seg_result.labeled_filled == prop.label
        )
        obj_area: int = int(np.count_nonzero(obj_mask_filled))

        obj_overlap: int = int(
            np.count_nonzero(obj_mask_filled & bw_selected)
        )
        overlap_fraction: float = float(obj_overlap / max(obj_area, 1))
        is_selected: bool = overlap_fraction >= overlap_threshold

        eroded_mask: NDArray[np.bool_] = cast(
            NDArray[np.bool_],
            binary_erosion(obj_mask_filled, footprint=erosion_footprint),
        )
        boundary_mask: NDArray[np.bool_] = obj_mask_filled & ~eroded_mask
        interior_mask: NDArray[np.bool_] = obj_mask_filled & ~eroded_mask

        boundary_vals: NDArray[np.float64] = roi_img[boundary_mask]
        interior_vals: NDArray[np.float64] = roi_img[interior_mask]

        boundary_grad_vals: NDArray[np.float64] = g_mag[boundary_mask]
        interior_grad_vals: NDArray[np.float64] = g_mag[interior_mask]

        perimeter: float = float(prop.perimeter)
        circularity: float = (
            (4.0 * np.pi * float(prop.area) / (perimeter**2))
            if perimeter > 0
            else 0.0
        )

        major_axis: float = float(prop.major_axis_length)
        minor_axis: float = float(prop.minor_axis_length)
        aspect_ratio: float = major_axis / max(minor_axis, eps)

        mean_b_intensity: float = (
            float(np.mean(boundary_vals))
            if boundary_vals.size > 0
            else float("nan")
        )
        mean_i_intensity: float = (
            float(np.mean(interior_vals))
            if interior_vals.size > 0
            else float("nan")
        )
        b_i_intensity_diff: float = mean_b_intensity - mean_i_intensity

        mean_b_grad: float = (
            float(np.mean(boundary_grad_vals))
            if boundary_grad_vals.size > 0
            else float("nan")
        )
        mean_i_grad: float = (
            float(np.mean(interior_grad_vals))
            if interior_grad_vals.size > 0
            else float("nan")
        )
        b_i_grad_diff: float = mean_b_grad - mean_i_grad
        b_i_grad_ratio: float = mean_b_grad / max(mean_i_grad, eps)

        obj_merged_area: int = int(
            np.count_nonzero(obj_mask_filled & seg_result.bw_merged)
        )
        hole_area: int = obj_area - obj_merged_area
        hole_area_fraction: float = float(hole_area / max(obj_area, 1))

        records.append(
            {
                "ImageName": roi_meta.image_name,
                "ROI_ID": roi_meta.roi_id,
                "ROI_Row": roi_meta.roi_row,
                "ROI_Col": roi_meta.roi_col,
                "ROI_X1": roi_meta.x1,
                "ROI_Y1": roi_meta.y1,
                "ROI_X2": roi_meta.x2,
                "ROI_Y2": roi_meta.y2,
                "ObjectLabel": k,
                "IsSelected": is_selected,
                "SelectionSource": selection_source,
                "OverlapFraction": overlap_fraction,
                "DarkThreshold": seg_result.params.dark_threshold,
                "AreaCutoff": seg_result.params.area_cutoff,
                "ConnectRadius": seg_result.params.connect_radius,
                "Solidity": float(prop.solidity),
                "Extent": float(prop.extent),
                "Circularity": circularity,
                "BoundaryInteriorGradRatio": b_i_grad_ratio,
                "BoundaryInteriorGradDiff": b_i_grad_diff,
                "MeanBoundaryGrad": mean_b_grad,
                "HoleAreaFraction": hole_area_fraction,
                "Eccentricity": float(prop.eccentricity),
                "AspectRatio": aspect_ratio,
                "MeanBoundaryIntensity": mean_b_intensity,
                "BoundaryInteriorIntensityDiff": b_i_intensity_diff,
                "Area": float(prop.area),
                "EquivDiameter": float(prop.equivalent_diameter_area),
            }
        )

    return pd.DataFrame(records)