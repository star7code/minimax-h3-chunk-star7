from __future__ import annotations

import comfy.sd
import comfy.utils
import folder_paths


INHERIT_FIRST_PASS = "继承一采"


def lora_choices() -> list[str]:
    return [INHERIT_FIRST_PASS, *folder_paths.get_filename_list("loras")]


def apply_selected_lora(owner, model, lora_name: str, strength: float = 1.0):
    selected = str(lora_name or INHERIT_FIRST_PASS)
    if selected == INHERIT_FIRST_PASS:
        return model, "inherit first pass"

    strength = float(strength)
    if strength == 0.0:
        return model, f"{selected} @ 0.00 (disabled)"

    available = folder_paths.get_filename_list("loras")
    if selected not in available:
        raise FileNotFoundError(
            f"Selected refinement LoRA is unavailable: {selected!r}. "
            "Place it in ComfyUI/models/loras and refresh the node list."
        )
    path = folder_paths.get_full_path_or_raise("loras", selected)
    cached = getattr(owner, "_star7_refine_lora", None)
    if cached is None or cached[0] != path:
        state, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True
        )
        cached = (path, state, metadata)
        owner._star7_refine_lora = cached
    patched, _clip = comfy.sd.load_lora_for_models(
        model, None, cached[1], strength, 0.0, lora_metadata=cached[2]
    )
    return patched, f"{selected} @ {strength:.2f}"
