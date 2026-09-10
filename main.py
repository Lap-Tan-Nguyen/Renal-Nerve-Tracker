import sys
from pathlib import Path

# 1. Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Dependency pre-flight check
from src.utils.setup_env import verify_and_install_requirements

verify_and_install_requirements()

# 3. Third-party imports
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import napari
from skimage.io import imread

# 4. Centralized config and module imports
import config
from src.gui.viewer import build_pipeline_viewer, display_full_slice


def ensure_directories() -> None:
    """Ensures local data and output directories exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_input_image_path() -> Optional[Path]:
    """Scans the data directory for supported histology slide images."""
    ensure_directories()
    available_files = [
        f for f in config.DATA_DIR.iterdir()
        if f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    if available_files:
        selected_file = available_files[0]
        print(f"[DATA] Found {len(available_files)} slide(s) in '{config.DATA_DIR}/'.")
        print(f"[DATA] Loading: {selected_file.name}")
        return selected_file

    print(
        f"[DATA] No supported images ({', '.join(config.SUPPORTED_EXTENSIONS)}) found in '{config.DATA_DIR}/'."
    )
    print(f"[DATA] Place your histology image file into ./{config.DATA_DIR}/ and re-run.")
    return None


def load_slide_image(file_path: Path) -> NDArray[np.float64]:
    """Loads and converts histology slide into normalized float grayscale data."""
    raw_img = imread(str(file_path))

    if raw_img.ndim == 3:
        if raw_img.shape[-1] in (3, 4):
            img_gray = raw_img[..., :3].mean(axis=2).astype(np.float64)
        else:
            img_gray = raw_img.mean(axis=0).astype(np.float64)
    else:
        img_gray = raw_img.astype(np.float64)

    max_val = float(np.max(img_gray))
    if max_val > 1.0:
        img_gray /= max_val

    return img_gray


def main() -> None:
    image_path = get_input_image_path()
    if image_path is None:
        sys.exit(0)

    img_full = load_slide_image(image_path)

    viewer = build_pipeline_viewer(
        title=f"{config.VIEWER_TITLE} - {image_path.name}",
        theme=config.VIEWER_THEME,
    )
    
    display_full_slice(
        viewer=viewer,
        img_full=img_full,
        layer_name=config.LAYER_FULL_IMAGE,
    )

    print("[APP] Starting Napari viewer event loop...")
    napari.run()


if __name__ == "__main__":
    main()