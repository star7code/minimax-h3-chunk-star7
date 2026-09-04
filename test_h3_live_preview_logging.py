import importlib.util
import io
import logging
import pathlib
import sys


COMFY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "h3_live_preview_logging_test",
    pathlib.Path(__file__).with_name("h3_live_preview.py"),
)
preview = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preview)


def test_preview_filter_hides_only_expected_info():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        with preview._quiet_preview_model_loading():
            logging.info("Requested to load TAEHV")
            logging.info("Model TAEHV prepared for dynamic VRAM loading. 43MB Staged.")
            logging.info("Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged.")
            logging.info("0 models unloaded.")
            logging.info("unrelated sampling information")
            logging.warning("Requested to load TAEHV")
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    text = output.getvalue()
    assert text.count("Requested to load TAEHV") == 1
    assert "unrelated sampling information" in text
    assert "prepared for dynamic VRAM loading" not in text
    assert "models unloaded" not in text


if __name__ == "__main__":
    test_preview_filter_hides_only_expected_info()
    print("H3 Live Preview logging tests passed")
