from typing import Any, List, Optional, Tuple, cast
import napari
from napari.layers import Image, Labels, Shapes
from napari.viewer import Viewer
import numpy as np
from numpy.typing import NDArray
import pandas as pd


def build_pipeline_viewer(
    title: str = "Histology Segmentation Pipeline",
    theme: str = "dark",
) -> Viewer:
    """Configures and returns a Napari viewer window instance."""
    viewer: Viewer = napari.Viewer(title=title, ndisplay=2)
    viewer.theme = theme
    return viewer


def display_full_slice(
    viewer: Viewer,
    img_full: NDArray[np.float64],
    layer_name: str = "Full Slide Image",
) -> Image:
    """Displays or updates the base histology slice on the viewer canvas[cite: 11]."""
    if layer_name in viewer.layers:
        img_layer = cast(Image, viewer.layers[layer_name])
        img_layer.data = cast(Any, img_full)
        return img_layer

    return cast(
        Image,
        cast(Any, viewer).add_image(
            img_full,
            name=layer_name,
            colormap="gray",
            blending="translucent",
        ),
    )


def add_or_update_labels(
    viewer: Viewer,
    data: NDArray[Any],
    layer_name: str,
    opacity: float = 0.7,
) -> Labels:
    """Safely updates an existing Labels layer in-place or creates a new layer."""
    if layer_name in viewer.layers:
        layer = cast(Labels, viewer.layers[layer_name])
        layer.data = cast(Any, data)
        layer.opacity = opacity
        return layer

    return cast(
        Labels,
        cast(Any, viewer).add_labels(
            data,
            name=layer_name,
            opacity=opacity,
        ),
    )


def focus_camera_on_roi(
    viewer: Viewer,
    bbox: Tuple[int, int, int, int],
    padding: float = 20.0,
) -> None:
    """Centers and zooms the viewer camera to a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    center_y = (y1 + y2) / 2.0
    center_x = (x1 + x2) / 2.0

    height = max(abs(y2 - y1) + (2 * padding), 1.0)

    view_size = viewer.window.geometry() if viewer.window else None
    canvas_height = float(view_size[3]) if view_size else 800.0
    zoom_level = canvas_height / height

    viewer.camera.center = (center_y, center_x)
    viewer.camera.zoom = zoom_level


def display_grid_overlay(
    viewer: Viewer,
    roi_table: pd.DataFrame,
    layer_name: str = "ROI Grid",
    active_roi_id: Optional[int] = None,
) -> Shapes:
    """Renders ROI grid bounding boxes from a DataFrame as a Shapes layer."""
    rectangles = []
    edge_colors = []

    for _, row in roi_table.iterrows():
        x1, y1, x2, y2 = int(row["X1"]), int(row["Y1"]), int(row["X2"]), int(row["Y2"])
        rect = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
        rectangles.append(rect)

        if active_roi_id is not None and int(row["ROI_ID"]) == active_roi_id:
            edge_colors.append("yellow")
        else:
            edge_colors.append("cyan")

    if layer_name in viewer.layers:
        viewer.layers.remove(viewer.layers[layer_name])

    return cast(
        Shapes,
        cast(Any, viewer).add_shapes(
            rectangles,
            shape_type="rectangle",
            name=layer_name,
            edge_color=edge_colors,
            face_color="transparent",
            edge_width=2,
        ),
    )


def safe_remove_layers(viewer: Viewer, layer_names: List[str]) -> None:
    """Removes specified layers from the viewer canvas if present."""
    for name in layer_names:
        if name in viewer.layers:
            viewer.layers.remove(viewer.layers[name])