from pathlib import Path
from typing import Set

# Application Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Supported File Formats
SUPPORTED_EXTENSIONS: Set[str] = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Napari Viewer Display Settings
VIEWER_TITLE: str = "Histology Segmentation Pipeline"
VIEWER_THEME: str = "dark"

# Centralized Napari Canvas Layer Names (Prevents string typo bugs)
LAYER_FULL_IMAGE: str = "Full Slide Image"
LAYER_ROI_GRID: str = "ROI Grid Overlay"
LAYER_ARTERY_MASK: str = "Artery Mask"
LAYER_NERVE_CANDIDATES: str = "Nerve Candidates"

# Default Grid & Processing Settings
DEFAULT_GRID_ROIS: int = 9
DEFAULT_CAMERA_PADDING: float = 20.0