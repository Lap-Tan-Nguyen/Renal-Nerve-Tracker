from typing import Any, Tuple, cast
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import disk, remove_small_objects
from skimage.morphology.binary import binary_closing, binary_dilation

from models.schemas import ArteryParams, ArteryResult


def keep_largest_component(bw: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Keep only the largest connected component in a binary mask[cite: 8]."""
    labeled_tuple = cast(Tuple[NDArray[np.int32], int], ndi.label(bw))
    labeled_mask, num_features = labeled_tuple

    if num_features == 0:
        return np.zeros_like(bw, dtype=bool)

    regions = regionprops(labeled_mask)
    largest_region = max(regions, key=lambda r: r.area)

    return cast(NDArray[np.bool_], labeled_mask == largest_region.label)


def segment_artery_from_roi(
    img_full: NDArray[np.float64],
    roi_pos: Tuple[int, int, int, int],
    params: ArteryParams,
) -> ArteryResult:
    """Segment main artery candidate from user-selected ROI."""
    if img_full.ndim == 3:
        img_full = img_full.mean(axis=2)

    x, y, w, h = roi_pos
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive.")

    # Crop ROI
    img_roi: NDArray[np.float64] = img_full[y : y + h, x : x + w]
    if img_roi.size == 0:
        raise ValueError("ROI crop failed. Check roi_pos coordinates.")

    # Preprocessing
    img_pre: NDArray[np.float64] = img_roi.copy()
    if params.use_clahe:
        img_pre = equalize_adapthist(
            img_pre, clip_limit=params.clahe_clip_limit
        )

    img_pre = gaussian(img_pre, sigma=params.gaussian_sigma)

    # Gradient Magnitude & Suppression
    g_mag: NDArray[np.float64] = sobel(img_pre)
    g_mag_supp: NDArray[np.float64] = g_mag.copy()
    g_mag_supp[g_mag_supp < params.grad_floor] = 0

    # Thresholding
    if params.use_otsu_threshold:
        t_base: float = float(threshold_otsu(g_mag_supp))
        thresh: float = t_base * params.otsu_scale_factor
        bw_edge: NDArray[np.bool_] = g_mag_supp > thresh
    else:
        raise NotImplementedError("Percentile thresholding not enabled.")

    # Cleanup & Closing
    bw_clean: NDArray[np.bool_] = cast(
        NDArray[np.bool_],
        remove_small_objects(bw_edge, min_size=params.min_blob_area),
    )
    raw_close_footprint: Any = cast(Any, disk)(params.close_radius)
    close_footprint: NDArray[np.uint8] = np.asarray(
        raw_close_footprint, dtype=np.uint8
    )
    bw_connected: NDArray[np.bool_] = cast(
        NDArray[np.bool_], binary_closing(bw_clean, footprint=close_footprint)
    )

    # Keep largest candidate
    bw_final: NDArray[np.bool_] = keep_largest_component(bw_connected)
    if not np.any(bw_final):
        raise RuntimeError("No final artery candidate found. Adjust parameters.")

    # Seal breaks & Fill lumen
    raw_seal_footprint: Any = cast(Any, disk)(params.fill_dilate_radius)
    seal_footprint: NDArray[np.uint8] = np.asarray(
        raw_seal_footprint, dtype=np.uint8
    )
    bw_seal: NDArray[np.bool_] = cast(
        NDArray[np.bool_], binary_dilation(bw_final, footprint=seal_footprint)
    )

    bw_filled_raw: NDArray[np.bool_] = cast(
        NDArray[np.bool_], ndi.binary_fill_holes(bw_seal)
    )
    bw_filled: NDArray[np.bool_] = keep_largest_component(bw_filled_raw)

    filled_gain: int = int(
        np.count_nonzero(bw_filled) - np.count_nonzero(bw_final)
    )

    # Centroid calculation
    props = regionprops(bw_filled.astype(np.int32))
    if not props:
        raise RuntimeError("Filled mask is empty. Could not compute centroid.")

    cy, cx = props[0].centroid
    centroid_roi: Tuple[float, float] = (float(cx), float(cy))

    # Map back to full image space
    mask_full: NDArray[np.bool_] = np.zeros_like(img_full, dtype=bool)
    mask_full[y : y + h, x : x + w] = bw_filled

    img_full_no_artery: NDArray[np.float64] = img_full.copy()
    img_full_no_artery[mask_full] = 0

    return ArteryResult(
        params=params,
        roi_pos=roi_pos,
        img_roi=img_roi,
        img_pre=img_pre,
        bw_final=bw_final,
        bw_filled=bw_filled,
        centroid_roi=centroid_roi,
        mask_full=mask_full,
        img_full_no_artery=img_full_no_artery,
        filled_gain=filled_gain,
    )