import importlib.util
import subprocess
import sys
from typing import Dict, List

# Maps module import names to PyPI package installation specs
REQUIRED_DEPENDENCIES: Dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "skimage": "scikit-image",
    "napari": "napari[all]",
    "magicgui": "magicgui",
    "pydantic": "pydantic",
}


def verify_and_install_requirements(
    requirements: Dict[str, str] = REQUIRED_DEPENDENCIES,
) -> None:
    """Checks for missing dependencies and automatically installs them via pip."""
    missing_packages: List[str] = []

    for import_name, pypi_spec in requirements.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(pypi_spec)

    if not missing_packages:
        print("[ENV] All required dependencies are installed and ready.")
        return

    print(
        f"[ENV] Missing required packages: {', '.join(missing_packages)}"
    )
    print("[ENV] Installing missing packages...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing_packages]
        )
        print("[ENV] All dependencies successfully installed.")
    except subprocess.CalledProcessError as err:
        print(
            f"[ENV ERROR] Package installation failed with code {err.returncode}.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    verify_and_install_requirements()