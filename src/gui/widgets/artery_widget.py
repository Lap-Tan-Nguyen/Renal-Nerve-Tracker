from typing import Any
from magicgui import magicgui
from src.models.schemas import ArteryParams


def create_artery_control_widget(controller: Any) -> Any:
    """Builds interactive GUI widget for tuning artery segmentation parameters[cite: 15, 19]."""

    @magicgui(
        call_button="Segment Artery from Selection",
        use_clahe={"label": "Use CLAHE"},
        clahe_clip_limit={"widget_type": "FloatSlider", "min": 0.001, "max": 0.05, "step": 0.001},
        gaussian_sigma={"widget_type": "FloatSlider", "min": 0.1, "max": 5.0, "step": 0.1},
        otsu_scale_factor={"widget_type": "FloatSlider", "min": 0.5, "max": 2.0, "step": 0.05},
        min_blob_area={"widget_type": "SpinBox", "min": 50, "max": 10000, "step": 50},
    )
    def widget(
        use_clahe: bool = True,
        clahe_clip_limit: float = 0.01,
        gaussian_sigma: float = 1.0,
        otsu_scale_factor: float = 1.15,
        min_blob_area: int = 800,
    ) -> None:
        params = ArteryParams(
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            gaussian_sigma=gaussian_sigma,
            otsu_scale_factor=otsu_scale_factor,
            min_blob_area=min_blob_area,
        )
        controller.run_auto_segmentation(params)

    @widget.append
    @magicgui(call_button="Save Artery Mask & Continue")
    def save_button() -> None:
        controller.save_artery_data()

    return widget