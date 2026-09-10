import os
from typing import Any, Optional, Tuple, cast
import napari
from napari.layers import Labels, Shapes
import numpy as np
from numpy.typing import NDArray

from engine.artery import segment_artery_from_roi
from models.schemas import ArteryParams, ArteryResult


class ArteryController:
    """Manages artery selection, automated segmentation, and mask creation."""

    def __init__(
        self,
        viewer: napari.Viewer,
        img_full: NDArray[np.float64],
        image_name: str,
        output_dir: str = "artery_segmentation_data",
    ) -> None:
        self.viewer: napari.Viewer = viewer
        self.img_full: NDArray[np.float64] = img_full
        self.image_name: str = image_name
        self.output_dir: str = output_dir

        self.last_result: Optional[ArteryResult] = None
        self.img_no_artery: NDArray[np.float64] = img_full.copy()

        self.roi_box_layer: Optional[Shapes] = None
        self.artery_mask_layer: Optional[Labels] = None

        self.setup_artery_layers()

    def setup_artery_layers(self) -> None:
        """Initializes shape and label layers for artery selection."""
        viewer_any: Any = self.viewer

        if "Full Slide" not in self.viewer.layers:
            viewer_any.add_image(self.img_full, name="Full Slide", colormap="gray")

        self.roi_box_layer = cast(
            Shapes,
            viewer_any.add_shapes(
                name="Artery ROI Box",
                shape_type="rectangle",
                edge_color="cyan",
                edge_width=3,
                face_color="transparent",
            ),
        )

        empty_mask: NDArray[np.int32] = np.zeros(
            (self.img_full.shape[0], self.img_full.shape[1]), dtype=np.int32
        )
        self.artery_mask_layer = cast(
            Labels,
            viewer_any.add_labels(empty_mask, name="Artery Mask", opacity=0.5),
        )

    def run_segmentation(self, params: ArteryParams) -> Optional[ArteryResult]:
        """Runs segmentation on user-drawn shape coordinates."""
        if self.roi_box_layer is None:
            print("No ROI selection layer available.")
            return None

        layer_data: Any = self.roi_box_layer.data
        if not layer_data or len(layer_data) == 0:
            print("Please draw a bounding box around the main artery first.")
            return None

        shape_coords: NDArray[np.float64] = np.asarray(
            layer_data[-1], dtype=np.float64
        )

        min_y: int = int(np.min(shape_coords[:, 0]))
        max_y: int = int(np.max(shape_coords[:, 0]))
        min_x: int = int(np.min(shape_coords[:, 1]))
        max_x: int = int(np.max(shape_coords[:, 1]))

        w: int = max_x - min_x
        h: int = max_y - min_y
        roi_pos: Tuple[int, int, int, int] = (min_x, min_y, w, h)

        self.last_result = segment_artery_from_roi(self.img_full, roi_pos, params)

        # Cast layer instance to Any to bypass LayerDataProtocol setter check
        if self.artery_mask_layer is not None:
            cast(Any, self.artery_mask_layer).data = self.last_result.mask_full.astype(
                np.int32
            )

        self.img_no_artery = self.last_result.img_full_no_artery
        return self.last_result

    def save_mask(self) -> str:
        """Saves artery mask data to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        base_name: str = os.path.splitext(self.image_name)[0]
        out_path: str = os.path.join(
            self.output_dir, f"{base_name}_artery_segmentation.npz"
        )

        mask_data: NDArray[np.bool_]
        if self.artery_mask_layer is not None:
            raw_data: Any = self.artery_mask_layer.data
            mask_data = cast(NDArray[np.bool_], np.asarray(raw_data) > 0)
        else:
            mask_data = np.zeros_like(self.img_full, dtype=bool)

        np.savez_compressed(
            out_path,
            image_name=self.image_name,
            mask_full=mask_data,
            img_full_no_artery=self.img_no_artery,
        )
        return out_path