import os
from typing import Any, List, Optional, Tuple, cast
import napari
from napari.layers import Image, Labels
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from engine.features import extract_nerve_candidate_features
from engine.nerve import segment_nerve_candidates_in_roi
from io.exporter import append_table_to_dataset, save_roi_reconstruction_data
from io.importer import find_completed_roi_ids
from models.schemas import NerveParams, NerveSegmentationResult, ROIMeta


class NerveController:
    """Manages ROI grid traversal, nerve candidate segmentation, and annotation export."""

    def __init__(
        self,
        viewer: napari.Viewer,
        img_full: NDArray[np.float64],
        image_name: str,
        roi_table: pd.DataFrame,
        dataset_path: str,
        reconstruction_dir: str,
    ) -> None:
        self.viewer: napari.Viewer = viewer
        self.img_full: NDArray[np.float64] = img_full
        self.image_name: str = image_name
        self.roi_table: pd.DataFrame = roi_table
        self.dataset_path: str = dataset_path
        self.reconstruction_dir: str = reconstruction_dir

        self.current_index: int = 0
        self.completed_ids: List[int] = find_completed_roi_ids(
            reconstruction_dir, image_name
        )

        # Canvas Layers
        self.img_layer: Optional[Image] = None
        self.candidate_layer: Optional[Labels] = None
        self.user_mask_layer: Optional[Labels] = None
        self.last_seg_result: Optional[NerveSegmentationResult] = None

    def get_current_roi_meta(self) -> ROIMeta:
        """Constructs metadata dataclass for the active ROI grid index."""
        row: pd.Series[Any] = self.roi_table.iloc[self.current_index]
        return ROIMeta(
            image_name=self.image_name,
            roi_id=int(row["ROI_ID"]),
            roi_row=int(row["ROI_Row"]),
            roi_col=int(row["ROI_Col"]),
            x1=int(row["X1"]),
            y1=int(row["Y1"]),
            x2=int(row["X2"]),
            y2=int(row["Y2"]),
        )

    def load_roi(self, params: NerveParams) -> None:
        """Crops ROI slice, performs candidate segmentation, and updates canvas layers."""
        meta: ROIMeta = self.get_current_roi_meta()
        roi_crop: NDArray[np.float64] = self.img_full[
            meta.y1 : meta.y2 + 1, meta.x1 : meta.x2 + 1
        ]

        # Execute candidate segmentation engine
        self.last_seg_result = segment_nerve_candidates_in_roi(roi_crop, params)

        # Clear previous ROI layers
        self.clear_roi_layers()

        viewer_any: Any = self.viewer

        # 1. Background ROI image slice
        self.img_layer = cast(
            Image,
            viewer_any.add_image(
                self.last_seg_result.roi_img, name="ROI Image", colormap="gray"
            ),
        )

        # 2. Automated dark candidate objects
        self.candidate_layer = cast(
            Labels,
            viewer_any.add_labels(
                self.last_seg_result.labeled_filled,
                name="Detected Candidates",
                opacity=0.4,
            ),
        )

        # 3. User-editable annotation mask canvas
        initial_user_mask: NDArray[np.int32] = (
            self.last_seg_result.bw_filled.astype(np.int32)
        )
        self.user_mask_layer = cast(
            Labels,
            viewer_any.add_labels(
                initial_user_mask, name="Edited Mask", opacity=0.5
            ),
        )

    def save_current_roi_and_advance(self, params: NerveParams) -> None:
        """Extracts morphological features, writes dataset row, and advances to next tile."""
        if self.last_seg_result is None or self.user_mask_layer is None:
            print("No ROI loaded or active user mask found.")
            return

        meta: ROIMeta = self.get_current_roi_meta()
        raw_mask_data: Any = self.user_mask_layer.data
        bw_selected: NDArray[np.bool_] = cast(
            NDArray[np.bool_], np.asarray(raw_mask_data) > 0
        )

        # Feature extraction and ground-truth dataset labeling
        features_df: pd.DataFrame = extract_nerve_candidate_features(
            roi_img=self.last_seg_result.roi_img,
            seg_result=self.last_seg_result,
            bw_selected=bw_selected,
            roi_meta=meta,
        )

        if not features_df.empty:
            append_table_to_dataset(features_df, self.dataset_path)

        # Save ROI reconstruction payload (.npz)
        save_roi_reconstruction_data(
            image_name=self.image_name,
            roi_meta=meta,
            roi_img=self.last_seg_result.roi_img,
            bw_selected=bw_selected,
            full_image_shape=(self.img_full.shape[0], self.img_full.shape[1]),
            out_folder=self.reconstruction_dir,
        )

        if meta.roi_id not in self.completed_ids:
            self.completed_ids.append(meta.roi_id)

        # Step navigation forward
        if self.current_index < len(self.roi_table) - 1:
            self.current_index += 1
            self.load_roi(params)
        else:
            print("Completed all ROIs in dataset session!")

    def previous_roi(self, params: NerveParams) -> None:
        """Navigates back to the previous ROI tile in the grid."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_roi(params)

    def clear_roi_layers(self) -> None:
        """Removes nerve-stage canvas layers from the Napari viewer."""
        for layer_name in ["ROI Image", "Detected Candidates", "Edited Mask"]:
            if layer_name in self.viewer.layers:
                layer_to_remove = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer_to_remove)