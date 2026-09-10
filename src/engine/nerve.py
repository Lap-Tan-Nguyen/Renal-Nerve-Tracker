from typing import Any, Optional, Tuple, cast
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from skimage.measure import label
from skimage.morphology import disk, remove_small_objects
from skimage.morphology.binary import binary_closing

from models.schemas import NerveParams, NerveSegmentationResult


def segment_nerve_candidates_in_roi(
    roi_img: NDArray[np.float64],
    params: Optional[NerveParams] = None,
) -> NerveSegmentationResult:
    """Segments dark nerve candidate objects within a single ROI tile."""
    if params is None:
        params = NerveParams()

    if roi_img.ndim == 3:
        roi_img_gray: NDArray[np.float64] = roi_img.mean(axis=2)
    else:
        roi_img_gray = roi_img.astype(np.float64)

    if float(np.max(roi_img_gray)) > 1.0:
        roi_img_gray = roi_img_gray / 255.0

    bw_dark: NDArray[np.bool_] = roi_img_gray < params.dark_threshold

    if not np.any(bw_dark):
        empty_mask: NDArray[np.bool_] = np.zeros_like(roi_img_gray, dtype=bool)
        empty_label: NDArray[np.int32] = np.zeros_like(roi_img_gray, dtype=np.int32)
        return NerveSegmentationResult(
            params=params,
            roi_img=roi_img_gray,
            bw_dark=empty_mask,
            bw_keep=empty_mask,
            bw_merged=empty_mask,
            bw_filled=empty_mask,
            labeled_filled=empty_label,
            num_objects=0,
        )

    bw_keep: NDArray[np.bool_] = cast(
        NDArray[np.bool_],
        remove_small_objects(bw_dark, min_size=params.area_cutoff),
    )

    raw_footprint: Any = cast(Any, disk)(params.connect_radius)
    footprint: NDArray[np.uint8] = np.asarray(raw_footprint, dtype=np.uint8)

    bw_merged: NDArray[np.bool_] = cast(
        NDArray[np.bool_], binary_closing(bw_keep, footprint=footprint)
    )

    bw_filled_raw: NDArray[np.bool_] = cast(
        NDArray[np.bool_], ndi.binary_fill_holes(bw_merged)
    )

    label_res: Tuple[NDArray[Any], int] = cast(
        Tuple[NDArray[Any], int], label(bw_filled_raw, return_num=True)
    )
    labeled_filled_raw, num_objects = label_res
    labeled_filled: NDArray[np.int32] = labeled_filled_raw.astype(np.int32)

    return NerveSegmentationResult(
        params=params,
        roi_img=roi_img_gray,
        bw_dark=bw_dark,
        bw_keep=bw_keep,
        bw_merged=bw_merged,
        bw_filled=bw_filled_raw,
        labeled_filled=labeled_filled,
        num_objects=int(num_objects),
    )