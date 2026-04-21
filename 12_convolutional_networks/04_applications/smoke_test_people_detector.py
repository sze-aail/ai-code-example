"""Tiny smoke test for 01_people_detection_camera.py without opening camera."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("01_people_detection_camera.py")


def load_module():
    spec = spec_from_file_location("people_detector", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {SCRIPT_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module()

    assert module.parse_source("0") == 0
    assert module.parse_source("1") == 1
    assert module.parse_source("video.mp4") == "video.mp4"

    print("smoke-test: ok")


if __name__ == "__main__":
    main()

