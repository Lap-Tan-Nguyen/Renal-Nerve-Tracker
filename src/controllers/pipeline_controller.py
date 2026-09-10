from enum import Enum, auto
from typing import Optional
import napari
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from controllers.artery_controller import ArteryController
from controllers.nerve_controller import NerveController
from engine.grid import build_auto_roi_grid


class PipelineStage(Enum):
    ARTERY_SEGMENTATION = auto()
    NERVE_ANNOTATION = auto()


class PipelineController:
    """Orchestrates stage transitions and coordinates sub-controllers."""

    def __init__(
        self,
        viewer: napari.Viewer,
        img_full: NDArray[np.float64],
        image_name: str,
        dataset_path: str,
        reconstruction_dir: str,
    ) -> None:
        self.viewer: napari.Viewer = viewer
        self.img_full: NDArray[np.float64] = img_full
        self.image_name: str = image_name
        self.dataset_path: str = dataset_path
        self.reconstruction_dir: str = reconstruction_dir

        self.current_stage: PipelineStage = PipelineStage.ARTERY_SEGMENTATION

        # Initialize Artery Stage
        self.artery_controller = ArteryController(
            viewer=self.viewer,
            img_full=self.img_full,
            image_name=self.image_name,
        )

        self.nerve_controller: Optional[NerveController] = None
        self.roi_table: Optional[pd.DataFrame] = None

    def transition_to_nerve_stage(self, num_rois: int = 9) -> None:
        """Completes artery removal, builds grid, and transitions to nerve annotation stage."""
        # Save artery results
        self.artery_controller.save_mask()
        cleaned_image: NDArray[np.float64] = self.artery_controller.img_no_artery

        # Build grid using cleaned image dimensions
        self.roi_table = build_auto_roi_grid(
            cleaned_image.shape, approx_num_rois=num_rois
        )

        # Remove artery layers to clear canvas
        for layer_name in ["Artery ROI Box", "Artery Mask"]:
            if layer_name in self.viewer.layers:
                layer_to_remove = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer_to_remove)

        # Initialize Nerve Controller with artery-suppressed slice
        self.nerve_controller = NerveController(
            viewer=self.viewer,
            img_full=cleaned_image,
            image_name=self.image_name,
            roi_table=self.roi_table,
            dataset_path=self.dataset_path,
            reconstruction_dir=self.reconstruction_dir,
        )

        self.current_stage = PipelineStage.NERVE_ANNOTATION
        print("Pipeline transitioned to Nerve Candidate Annotation Stage.")