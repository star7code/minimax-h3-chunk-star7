import importlib.util
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(ROOT.parents[1])
    folder_paths.models_dir = str(ROOT.parents[1] / "models")
    folder_paths.get_temp_directory = lambda: str(ROOT / ".test-temp")
    sys.modules.setdefault("folder_paths", folder_paths)
    spec = importlib.util.spec_from_file_location("star7_dlss_test_module", ROOT / "dlss_neural_enhance.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_target_megapixels_never_downscales_and_preserves_aspect():
    module = _load_module()
    assert module._target_size(1281, 721, 0.5) == (1282, 722, 0.925604, True)
    width, height, megapixels, preserved = module._target_size(1280, 720, 2.0)
    assert (width, height, preserved) == (1886, 1062, False)
    assert abs(megapixels - 2.0) < 0.01


def test_preset_overrides_custom_controls_only_outside_custom():
    module = _load_module()
    assert "真实风格加强" not in module._PRESET_NAMES
    assert "真实风格加强" in module.Star7DLSSNeuralEnhance.INPUT_TYPES()["required"]["风格预设"][0]
    preset = module._settings("真实人像优化", 0.1, 0.2, 0.3, 0.4, 0.2, False)
    assert preset["strength"] == 0.85
    assert preset["style"] == 1
    assert preset["skin"] == 0.0
    assert preset["mask"] is True
    custom = module._settings("自定义参数", 0.1, 0.2, 0.3, 0.4, 0.2, False)
    assert custom == {
        "style": 1,
        "strength": 0.1,
        "tone": 0.3,
        "structure": 0.2,
        "skin": 0.4,
        "temporal": 0.2,
        "mask": False,
    }


def test_lanczos_resize_preserves_channels_and_target_size():
    module = _load_module()
    import numpy as np

    frame = np.zeros((32, 48, 3), dtype=np.float32)
    resized = module._resize_lanczos(frame, 96, 64)
    assert resized.shape == (64, 96, 3)
    assert resized.dtype == np.float32


def test_temporal_stabilizer_ignores_motion_and_smooths_static_residual():
    module = _load_module()
    import numpy as np

    source = np.zeros((8, 8, 3), dtype=np.float32)
    previous_residual = np.full_like(source, 0.1)
    processed = np.full_like(source, 0.2)
    stable, residual = module._stabilize_residual(source, processed, source, previous_residual, 0.5)
    assert np.allclose(stable, 0.15)
    moved_source = np.ones_like(source)
    moved, _ = module._stabilize_residual(moved_source, processed, source, previous_residual, 0.5)
    assert np.allclose(moved, processed)


def test_download_uses_atomic_candidate_and_fallback_sources():
    module = _load_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.dll"
        source.write_bytes(b"MZtest-model")
        destination = root / "models" / "nvngx_dlssnr.dll"
        original = (
            module._MODEL, module._MODEL_SOURCES, module._validate_model,
            module._DOWNLOAD_ERROR,
        )
        try:
            module._MODEL = destination
            module._MODEL_SOURCES = (
                ("broken", (root / "missing.dll").as_uri(), False),
                ("local-test", source.as_uri(), False),
            )
            module._validate_model = lambda path, require_known_hash: "test-sha"
            module._DOWNLOAD_ERROR = None
            module._DOWNLOAD_DONE.clear()
            module._download_model_worker()
            assert destination.read_bytes() == b"MZtest-model"
            assert module._DOWNLOAD_ERROR is None
            assert not list(destination.parent.glob("*.download"))
            assert not list(destination.parent.glob("*.candidate"))
        finally:
            module._MODEL, module._MODEL_SOURCES, module._validate_model, module._DOWNLOAD_ERROR = original
