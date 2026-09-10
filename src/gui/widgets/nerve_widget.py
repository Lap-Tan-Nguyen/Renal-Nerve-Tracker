from typing import Any, List, Protocol, cast
from magicgui.widgets import Container, Label, PushButton
from magicgui import magicgui

from models.schemas import NerveParams, ROIMeta


class NerveControllerProtocol(Protocol):
    """Protocol defining the interface required from NervePipelineController for type safety."""

    current_index: int
    completed_ids: List[int]
    roi_table: Any

    def get_current_roi_meta(self) -> ROIMeta: ...
    def load_roi(self, params: NerveParams) -> None: ...
    def save_current_roi_and_advance(self, params: NerveParams) -> None: ...


def create_nerve_control_widget(
    controller: NerveControllerProtocol,
) -> Container:
    """Builds interactive GUI panel with live parameter controls, status readout, and step navigation[cite: 3, 16, 20]."""
    
    status_label = Label(value="ROI: -- / -- | Status: Ready")

    def update_status_text() -> None:
        """Updates UI status label with current ROI step and completion tracking[cite: 3]."""
        total_rois: int = len(controller.roi_table)
        meta: ROIMeta = controller.get_current_roi_meta()
        is_done: str = "✓ Completed" if meta.roi_id in controller.completed_ids else "Pending"
        status_label.value = (
            f"ROI {meta.roi_id} ({controller.current_index + 1}/{total_rois}) "
            f"[Row {meta.roi_row}, Col {meta.roi_col}] | {is_done}"
        )

    @magicgui(
        call_button="Process / Refresh ROI",
        dark_threshold={"widget_type": "FloatSlider", "min": 0.05, "max": 0.95, "step": 0.01, "label": "Dark Threshold"},
        area_cutoff={"widget_type": "SpinBox", "min": 5, "max": 5000, "step": 5, "label": "Min Area Cutoff"},
        connect_radius={"widget_type": "SpinBox", "min": 1, "max": 10, "step": 1, "label": "Connect Radius"},
    )
    def param_widget(
        dark_threshold: float = 0.35,
        area_cutoff: int = 50,
        connect_radius: int = 2,
    ) -> None:
        params = NerveParams(
            dark_threshold=dark_threshold,
            area_cutoff=area_cutoff,
            connect_radius=connect_radius,
        )
        controller.load_roi(params)
        update_status_text()

    @magicgui(call_button="Save ROI & Advance ->")
    def save_widget() -> None:
        params = NerveParams(
            dark_threshold=float(param_widget.dark_threshold.value),
            area_cutoff=int(param_widget.area_cutoff.value),
            connect_radius=int(param_widget.connect_radius.value),
        )
        controller.save_current_roi_and_advance(params)
        update_status_text()

    @magicgui(call_button="<- Previous ROI")
    def prev_widget() -> None:
        if controller.current_index > 0:
            controller.current_index -= 1
            params = NerveParams(
                dark_threshold=float(param_widget.dark_threshold.value),
                area_cutoff=int(param_widget.area_cutoff.value),
                connect_radius=int(param_widget.connect_radius.value),
            )
            controller.load_roi(params)
            update_status_text()

    # Initial status refresh
    update_status_text()

    # Assemble into unified side-panel layout
    panel = Container(
        widgets=[
            status_label,
            param_widget,
            save_widget,
            prev_widget,
        ]
    )

    return panel