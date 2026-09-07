"""Prepare a hosted Kaggle/Colab runtime without resolving the full uv lockfile."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
from collections.abc import Iterable

BASE_PACKAGES = (
    ("ipywidgets", "ipywidgets"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("yaml", "pyyaml"),
)
VISION_PACKAGES = (
    ("ultralytics", "ultralytics"),
    ("ultralytics_thop", "ultralytics-thop"),
)
OCR_PACKAGES = (("pytesseract", "pytesseract"),)


def _pip_install(packages: Iterable[str], *, no_deps: bool = True) -> None:
    package_list = list(dict.fromkeys(packages))
    if not package_list:
        return
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "-q",
    ]
    if no_deps:
        command.append("--no-deps")
    command.extend(package_list)
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def _missing_packages(specs: Iterable[tuple[str, str]]) -> list[str]:
    return [
        distribution
        for module, distribution in specs
        if importlib.util.find_spec(module) is None
    ]


def _ensure_packages(specs: Iterable[tuple[str, str]]) -> None:
    missing = _missing_packages(specs)
    if missing:
        _pip_install(missing)


def _roboflow_dependencies() -> list[str]:
    requirements = importlib.metadata.requires("roboflow") or []
    names: list[str] = []
    for requirement in requirements:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
        if match:
            names.append(match.group(1))
    return names


def _ensure_roboflow() -> None:
    _ensure_packages((("roboflow", "roboflow"),))
    # Install Roboflow's direct dependencies without invoking pip's resolver.
    # This avoids pulling a second PyTorch/CUDA stack into the hosted runtime.
    _pip_install(_roboflow_dependencies())


def _verify_modules(modules: Iterable[str]) -> None:
    failures: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"{module}: {error}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(f"Hosted runtime dependency check failed:\n{details}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "inference"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is missing from the hosted runtime; enable Kaggle GPU and restart the session"
        )
    if importlib.util.find_spec("torchvision") is None:
        raise RuntimeError(
            "Torchvision is missing from the hosted runtime; "
            "enable Kaggle GPU and restart the session"
        )
    _ensure_packages(BASE_PACKAGES)
    _ensure_packages(VISION_PACKAGES)
    _ensure_packages(OCR_PACKAGES)
    required_modules = [
        "numpy",
        "cv2",
        "yaml",
        "torch",
        "torchvision",
        "ultralytics",
        "ultralytics_thop",
        "pytesseract",
    ]
    if args.mode == "train":
        _ensure_roboflow()
        required_modules.append("roboflow")
    _verify_modules(required_modules)
    print(f"Hosted runtime ready for {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
