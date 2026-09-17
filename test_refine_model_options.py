from . import refine_model_options


def test_inherit_first_pass_keeps_the_exact_model():
    model = object()
    output, summary = refine_model_options.apply_selected_lora(
        object(), model, refine_model_options.INHERIT_FIRST_PASS
    )
    assert output is model
    assert summary == "inherit first pass"


def test_selected_lora_is_loaded_once_and_applied_model_only(monkeypatch):
    loads = []
    applications = []
    patched = object()

    monkeypatch.setattr(
        refine_model_options.folder_paths,
        "get_filename_list",
        lambda kind: ["repair.safetensors"] if kind == "loras" else [],
    )
    monkeypatch.setattr(
        refine_model_options.folder_paths,
        "get_full_path_or_raise",
        lambda kind, name: f"D:/models/{kind}/{name}",
    )
    monkeypatch.setattr(
        refine_model_options.comfy.utils,
        "load_torch_file",
        lambda path, **kwargs: (loads.append(path) or {"weight": 1}, {"meta": 1}),
    )

    def apply(model, clip, state, strength_model, strength_clip, lora_metadata=None):
        applications.append(
            (model, clip, state, strength_model, strength_clip, lora_metadata)
        )
        return patched, clip

    monkeypatch.setattr(refine_model_options.comfy.sd, "load_lora_for_models", apply)
    owner = type("Owner", (), {})()
    source = object()
    assert refine_model_options.apply_selected_lora(
        owner, source, "repair.safetensors", 0.65
    ) == (patched, "repair.safetensors @ 0.65")
    assert refine_model_options.apply_selected_lora(
        owner, source, "repair.safetensors", -0.25
    ) == (patched, "repair.safetensors @ -0.25")
    assert len(loads) == 1
    assert len(applications) == 2
    assert all(item[1] is None for item in applications)
    assert applications[0][3:] == (0.65, 0.0, {"meta": 1})
    assert applications[1][3:] == (-0.25, 0.0, {"meta": 1})


def test_zero_strength_skips_loading(monkeypatch):
    monkeypatch.setattr(
        refine_model_options.comfy.utils,
        "load_torch_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    model = object()
    output, summary = refine_model_options.apply_selected_lora(
        object(), model, "repair.safetensors", 0.0
    )
    assert output is model
    assert summary == "repair.safetensors @ 0.00 (disabled)"
