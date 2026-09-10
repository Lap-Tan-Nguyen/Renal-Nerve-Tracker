from typing import Any
from magicgui import magicgui


def create_export_widget(dataset_controller: Any, image_name: str, img_full: Any) -> Any:
    """Builds interactive GUI widget for monitoring annotated stats and exporting overlays[cite: 1, 14]."""

    @magicgui(call_button="Show Summary Stats")
    def widget() -> None:
        stats = dataset_controller.get_dataset_summary()
        print("=== Dataset Annotation Summary ===")
        for key, val in stats.items():
            print(f"{key}: {val}")

    @widget.append
    @magicgui(call_button="Export High-Res TIFF Overlay")
    def export_button() -> None:
        dataset_controller.export_overlay_tiff(image_name, img_full)

    return widget